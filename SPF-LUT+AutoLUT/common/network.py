import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############

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

class MuLUTConvUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, out_c=1, dense=True):
        super(MuLUTConvUnit, self).__init__()
        self.act = nn.ReLU()
        self.residual=Residual((2,2))

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x, prev_x):
        if prev_x!=None:
            x = self.residual(x, prev_x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class MuLUTConv(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, sample_size, nf=64, out_c=None, dense=True, stride=1):
        super(MuLUTConv, self).__init__()
        self.mode = mode
        self.sampler=AutoSample(sample_size)

        self.model = MuLUTConvUnit('2x2', nf, out_c=out_c, dense=dense)
        self.K = sample_size
        self.P = self.K-1
        self.S = 1  # PixelShuffle is in ConvBlock, we don't upscale here

        self.stride = stride

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

class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, in_c, out_c, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '1x1':
            self.conv1 = Conv(in_c, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, out_c, 1)

    def forward(self, x, prev_x="Unused"):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x



if __name__ == '__main__':
    pass
