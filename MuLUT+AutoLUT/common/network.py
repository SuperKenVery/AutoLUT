import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from loguru import logger
from rich import inspect


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))

class LrLambda:
    def __init__(self, opt):
        self.set5_psnr=0
        self.opt=opt
    def __call__(self,i):
        if self.set5_psnr<20:
            self.opt.valStep=10
            return 10
        elif self.set5_psnr<24:
            self.opt.valStep=20
            return 1
        elif self.set5_psnr<28.1:
            self.opt.valStep=30
            return 0.5
        elif self.set5_psnr<30:
            self.opt.valStep=50
            return 0.1
        else:
            return 0.05

############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, act='relu'):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        if act=='relu':
            self.act = nn.ReLU()
        elif act=='gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64, act='relu'):
        super(DenseConv, self).__init__()
        if act=='relu':
            self.act = nn.ReLU()
        elif act=='gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out

############### Other tools ###############

class Residual(nn.Module):
    def __init__(self, input_shape):
        assert len(input_shape)==2
        super().__init__()
        self.shape=input_shape
        self.weights=nn.Parameter(torch.zeros(self.shape))

    def forward(self, x, prev_x):
        assert x.shape[-2:]==self.shape and prev_x.shape[-2:]==self.shape
        with torch.no_grad():
            self.weights.data=torch.clamp(self.weights,0,1)

        averaged=self.weights*prev_x+(1-self.weights)*x

        return averaged

class AutoSample(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_shape=input_size
        self.sampler=nn.Conv2d(1,4,input_size)
        self.shuffel=nn.PixelShuffle(2)
        self.nw=input_size**2

    def forward(self, x):
        assert len(x.shape)==4 and x.shape[-2:]==(self.input_shape,self.input_shape), f"Unexpected shape: {x.shape}"
        # x = self.sampler(x)
        # logger.debug(self.sampler.weight)
        w = F.softmax(self.sampler.weight.view(-1, self.nw), dim=1).view_as(self.sampler.weight)
        x = F.conv2d(x, w, bias=self.sampler.bias*0)
        x = self.shuffel(x)
        return x

############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True, residual=False, act='relu'):
        super(MuLUTUnit, self).__init__()
        if act=='relu':
            self.act = nn.ReLU()
        elif act=='gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")
        self.upscale = upscale
        self.has_residual=residual

        if mode == '2x2':
            self.input_shape = (2,2)
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.input_shape = (3,3)
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.input_shape = (4,4)
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.input_shape = (1,4)
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if residual:
            self.residual = Residual(self.input_shape)

        if dense:
            self.conv2 = DenseConv(nf, nf, act=act)
            self.conv3 = DenseConv(nf + nf * 1, nf, act=act)
            self.conv4 = DenseConv(nf + nf * 2, nf, act=act)
            self.conv5 = DenseConv(nf + nf * 3, nf, act=act)
            self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1, act=act)
            self.conv3 = ActConv(nf, nf, 1, act=act)
            self.conv4 = ActConv(nf, nf, 1, act=act)
            self.conv5 = ActConv(nf, nf, 1, act=act)
            self.conv6 = Conv(nf, upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x, prev_x=None):
        if self.has_residual:
            x = self.residual(x, prev_x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x


class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, mode, nf, act='relu'):
        super(MuLUTcUnit, self).__init__()
        if act=='relu':
            self.act = nn.ReLU()
        elif act=='gelu':
            self.act = nn.GELU()
        else:
            raise AttributeError(f"Unknown activate function: {act}")

        if mode == '1x1':
            self.conv1 = Conv(3, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, 3, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        return x


############### Image Super-Resolution ###############
class SRNet(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, sample_size, nf=64, upscale=1, dense=True, residual=False, act='relu'):
        super(SRNet, self).__init__()
        self.residual = residual

        self.K = sample_size
        self.S = upscale

        self.sampler = AutoSample(sample_size)
        self.model = MuLUTUnit('2x2', nf, upscale, dense=dense, residual=residual, act=act)

        self.P = self.K - 1

    def unfold(self, x):
        """
        Do the convolution sampling
        """
        if x is None: return x, None
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        return x, (B, C, H, W)

    def put_back(self, x, ori_shape):
        B, C, H, W=ori_shape
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


    def forward(self, x, prev_x=None):
        # Here, prev_x is unfolded multiple times (previously unfolded as x)
        # TODO: Maybe we can do a speedup here
        # logger.debug(f"SRNet got {x.shape}")
        x, shape=self.unfold(x)
        prev_x, _=self.unfold(prev_x)

        x = self.sampler(x)
        # logger.debug(f"after sample {x}")
        if prev_x is not None:
            prev_x = self.sampler(prev_x)

        x = self.model(x, prev_x)   # B*C*L,K,K
        # logger.debug(f"shape after model: {x.shape}")

        x=self.put_back(x, shape)

        return x


############### Grayscale Denoising, Deblocking, Color Image Denosing ###############
class DNNet(nn.Module):
    """ Wrapper of basic MuLUT block without upsampling. """

    def __init__(self, mode, nf=64, dense=True):
        super(DNNet, self).__init__()
        self.mode = mode

        self.S = 1
        if mode == 'Sx1':
            self.model = MuLUTUnit('2x2', nf, dense=dense)
            self.K = 2
        elif mode == 'Dx1':
            self.model = MuLUTUnit('2x2d', nf, dense=dense)
            self.K = 3
        elif mode == 'Yx1':
            self.model = MuLUTUnit('1x4', nf, dense=dense)
            self.K = 3
        else:
            raise AttributeError
        self.P = self.K - 1

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)   # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))     # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x


############### Image Demosaicking ###############
class DMNet(nn.Module):
    """ Wrapper of the first stage of MuLUT network for demosaicking. 4D(RGGB) bayer patter to (4*3)RGB"""

    def __init__(self, mode, nf=64, dense=False):
        super(DMNet, self).__init__()
        self.mode = mode

        if mode == 'SxN':
            self.model = MuLUTUnit('2x2', nf, upscale=2, out_c=3, dense=dense)
            self.K = 2
            self.C = 3
        else:
            raise AttributeError
        self.P = 0  # no need to add padding self.K - 1
        self.S = 2  # upscale=2, stride=2

    def forward(self, x):
        B, C, H, W = x.shape
        # bayer pattern, stride = 2
        x = F.unfold(x, self.K, stride=2)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H // 2) * (W // 2))  # stride = 2
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H // 2) * (W // 2),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        # print("in", torch.round(x[0, 0]*255))

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)

        x = self.model(x)  # B*C*L,out_C,S,S
        # self.C along with feat scale
        x = x.reshape(B, C, (H // 2) * (W // 2), -1)  # B,C,L,out_C*S*S
        x = x.permute((0, 1, 3, 2))  # B,C,outC*S*S,L
        x = x.reshape(B, -1, (H // 2) * (W // 2))  # B,C*out_C*S*S,L
        x = F.fold(x, ((H // 2) * self.S, (W // 2) * self.S),
                   self.S, stride=self.S)
        return x
