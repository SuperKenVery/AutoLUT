import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "../")  # run under the current directory
from common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}

def identity(input):
    return input

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

class ConvBlock(nn.Module):
    def __init__(self, in_c, modes=['s', 'd', 'y'], nf=64):
        super(ConvBlock, self).__init__()
        self.in_c = in_c
        self.modes = modes
        self.module_dict = dict()

        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = MuLUTConv('{}x{}'.format(mode.upper(), 'N'),
                                                                                    nf=nf, out_c=1,stride=1)
        self.module_dict = nn.ModuleDict(self.module_dict)

    def forward(self, x):
        modes = self.modes

        x_out = []
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            pred = 0
            for mode in modes:
                pad = mode_pad_dict[mode]
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.tanh(torch.rot90(
                        sub_module(F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate')),
                        (4 - r) % 4, [2, 3])) * 127)

            x_out.append(pred)

        x = torch.cat(x_out,dim=1)

        return x

class ConvBlockV2(nn.Module):
    def __init__(self, in_c, out_c, scale=None, output_quant=False, modes=['s', 'd', 'y'], nf=64):
        super(ConvBlockV2, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.module_dict = dict()
        self.upscale = scale
        self.output_quant = output_quant

        scale_factor = 1 if scale is None else scale ** 2
        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = MuLUTConv('{}x{}'.format(mode.upper(), 'N'),
                                                                                    nf=nf, out_c=out_c * scale_factor,
                                                                                    stride=1)
        self.module_dict = nn.ModuleDict(self.module_dict)

    def forward(self, x):
        modes = self.modes

        x_out = 0
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            pred = 0
            for mode in modes:
                pad = mode_pad_dict[mode]
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.tanh(torch.rot90(
                        sub_module(F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate')),
                        (4 - r) % 4, [2, 3])) * 127)

            x_out += pred
        if self.output_quant:
            avg_factor = len(modes) * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            # avg_factor = len(modes)
            # x = x_out / avg_factor
            x = x_out / self.in_c

        return x
    
class BaseDNNets(nn.Module):
    def __init__(self, nf=64, modes=['s', 'd', 'y'], stages=2):
        super(BaseDNNets, self).__init__()
        self.modes = modes
        self.stages = stages

        self.module_list = []
        for s in range(stages):
            module = ConvBlock(in_c=1,modes=modes,nf=nf)
            self.module_list.append(module)

        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = torch.clamp(x, 0, 1)
        x = x.reshape((B*C,1,H,W))

        for s in range(self.stages):
            x = self.module_list[s](x)
            if s + 1 == self.stages:

                avg_factor, bias, norm = len(self.modes), 0, 1
                x = round_func((x / avg_factor) + bias)
                if phase == "train":
                    x = x / 255.0
            else:
                avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
                x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        x = x.reshape((B,C,H,W))
        return x

class SPF_LUT_net(nn.Module):
    def __init__(self, nf=32, modes=['s', 'd', 'y'], stages=2):
        super(SPF_LUT_net, self).__init__()
        self.modes = modes

        self.convblock1 = ConvBlockV2(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock2 = ConvBlockV2(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock3 = ConvBlockV2(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock4 = ConvBlockV2(1, 1, scale=None, output_quant=False, modes=modes, nf=nf)
        self.ChannelConv = MuLUTcUnit(in_c=4, out_c=4, mode='1x1', nf=nf)
        self.upblock = ConvBlockV2(4, 1, scale=None, output_quant=False, modes=modes, nf=nf)


    def forward(self, x, phase='train'):
        B, C, H, W = x.size()
        x = torch.clamp(x, 0, 1)
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        # block1
        x = self.convblock1(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block2
        x = self.convblock2(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block3
        x = self.convblock3(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block4
        x = self.convblock4(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        x = self.upblock(x)
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, H, W))

        return x
    
# lut_folder, stages, modes, lutName, sigma, interval, compressed_dimensions, diagonal_width, sampling_interval, c_in = 1
class BaseMuLUT_DFC(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, sigma, interval, compressed_dimensions, diagonal_width, sampling_interval, c_in=3):
        super(BaseMuLUT_DFC, self).__init__()
        self.interval = interval
        self.sigma = sigma
        self.modes = modes
        self.stages = stages
        self.c_in = c_in
        self.d = diagonal_width
        self.compression = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1

        # if compressed_dimensions in ['xyz', 'xyzt']:
        #     self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_L{}_d{}.npy'.format(L, diagonal_width)))
        #     self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        # else:
        #     self.ref2index = None
        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for s in range(stages):
            stage = s + 1
            for c in range(c_in):
                for mode in modes:
                    lut_path = os.path.join(lut_folder,
                                            "{}_s{}_c{}_{}_compress1.npy".format(lutName, str(stage), str(c), mode))
                    key = "s{}_c{}_{}_compress1".format(str(stage), str(c), mode)
                    if self.compression=='xy':
                        lut_arr = np.load(lut_path).reshape(-1, L * L, c_in).astype(np.float32) / 127.0
                    elif self.compression=='xyz':
                        lut_arr = np.load(lut_path).reshape(-1, L, c_in).astype(np.float32) / 127.0
                    elif self.compression=='xyzt':
                        lut_arr = np.load(lut_path).reshape(-1, c_in).astype(np.float32) / 127.0
                    else:
                        raise ValueError("compression {} not implemented.".format(self.compression))
                    self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                    lut_path = os.path.join(lut_folder,
                                            "{}_s{}_c{}_{}_compress2.npy".format(lutName, str(stage), str(c), mode))
                    key = "s{}_c{}_{}_compress2".format(str(stage), str(c), mode)
                    lut_arr = np.load(lut_path).reshape(-1, c_in).astype(np.float32) / 127.0
                    self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    @staticmethod
    def get_1d_index(i, j, N, L, diag):
        d = (diag - 1) // 2
        if 2 * d < L + 3:
            specialRowNum = (diag - 5) // 2

            index1 = (i >= 0) & (i <= specialRowNum)
            index2 = (i > specialRowNum) & (i < L - specialRowNum - 1)
            index3 = (i >= L - specialRowNum - 1) & (i <= L - 1)

            k = torch.floor_divide(i * (i + diag), 2).type(torch.int64) + j
            k[index2] = (diag - 1) * i[index2] + j[index2] + ((5 - diag) * (1 + diag)) // 8 - 1
            k[index3] = N - torch.floor_divide((L - i[index3] - 1) * (L - i[index3] - 1 + diag), 2).type(
                torch.int64) - L + j[index3]
        elif 1 + d <= L:
            specialRowNum = L - (d + 1) - 1
            index1 = (i >= 0) & (i <= specialRowNum)
            index2 = (i > specialRowNum) & (i < L - specialRowNum - 1)
            index3 = (i >= L - specialRowNum - 1) & (i <= L - 1)

            k = i * (1 + d) + torch.floor_divide(i * (i - 1), 2).type(torch.int64) + j
            k[index2] = (i[index2] - specialRowNum - 1) * L + j[index2] + (
                    (specialRowNum + 1) * (2 * L - specialRowNum - 2)) // 2
            k[index3] = N - ((L - i[index3] - 1) * (1 + d) + torch.floor_divide(
                (L - i[index3] - 1) * (L - 1 - i[index3] - 1), 2).type(torch.int64) + L - j[index3])
        return k

    def InterpTorchBatch_compress1(self, weight_c1, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        # img_x = img_in[:, :, 0:0 + h, 0:0 + w] / float(q)
        # img_y = img_in[:, :, 0:0 + h, 1:1 + w] / float(q)
        # index_flag = (torch.abs(img_x - img_y) <= self.d)

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
            fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 2:2 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 2:2 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 2:2 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
            fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
            fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 2:2 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
            fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.get_1d_index(img_a1, img_b1, N, L, diag)
        k01 = k00 + 1
        k10 = self.get_1d_index(img_a2, img_b1, N, L, diag)
        k11 = k10 + 1

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, self.c_in, 1))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, self.c_in, 1))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, self.c_in, 1))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, self.c_in, 1))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, self.c_in, 1))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, self.c_in, 1))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, self.c_in, 1))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, self.c_in, 1))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, self.c_in, 1))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, self.c_in, 1))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, self.c_in, 1))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, self.c_in, 1))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, self.c_in, 1))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, self.c_in, 1))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, self.c_in, 1))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, self.c_in, 1))

        out = torch.zeros((img_a1.shape[0], self.c_in, 1), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, self.c_in))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]


        p0000 = weight_c1[k0000].reshape((-1, out_c))
        p0001 = weight_c1[k0001].reshape((-1, out_c))
        p0010 = weight_c1[k0010].reshape((-1, out_c))
        p0011 = weight_c1[k0011].reshape((-1, out_c))
        p0100 = weight_c1[k0100].reshape((-1, out_c))
        p0101 = weight_c1[k0101].reshape((-1, out_c))
        p0110 = weight_c1[k0110].reshape((-1, out_c))
        p0111 = weight_c1[k0111].reshape((-1, out_c))

        p1000 = weight_c1[k1000].reshape((-1, out_c))
        p1001 = weight_c1[k1001].reshape((-1, out_c))
        p1010 = weight_c1[k1010].reshape((-1, out_c))
        p1011 = weight_c1[k1011].reshape((-1, out_c))
        p1100 = weight_c1[k1100].reshape((-1, out_c))
        p1101 = weight_c1[k1101].reshape((-1, out_c))
        p1110 = weight_c1[k1110].reshape((-1, out_c))
        p1111 = weight_c1[k1111].reshape((-1, out_c))

        out = torch.zeros((img_a1.shape[0], out_c), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c))

        out = out / q
        return out, index_flag

    def InterpTorchBatch(self, weight_c1, weight_c2, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1(weight_c1, mode, img_in, bd)
        elif self.compression == 'xyz':
            out_compress1, index_flag = None, None
        else:
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, self.c_in, mode, img_in, bd)
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval 
        L = 2 ** (8 - interval) + 1

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
            fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 2:2 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 2:2 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 2:2 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
            fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
            fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 2:2 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
            fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], self.c_in), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, self.c_in))
            out_all = out_all.permute(0, 1, 4, 2, 3).reshape((sz0, sz1 * self.c_in, sz2, sz3))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, self.c_in, 1))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, self.c_in, 1))



        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0,sz1,sz2,sz3, self.c_in, 1))
        out_all = out_all.permute(0, 1, 4, 2, 3, 5).reshape(
            (sz0, sz1 * self.c_in, sz2, sz3))
        return out_all

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        if self.c_in == 1:
            x = x.reshape((B * C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        modes, stages = self.modes, self.stages
        for s in range(stages):
            pred = 0
            stage = s + 1
            if stage == stages:
                avg_factor, bias, norm = len(modes), 0, 1
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
            for c in range(self.c_in):
                x_c = x[:, c:c + 1, :, :]
                for mode in modes:
                    pad = mode_pad_dict[mode]
                    key = "s{}_c{}_{}_compress1".format(str(stage), str(c), mode)
                    weight_c1 = getattr(self, "weight_" + key)
                    key = "s{}_c{}_{}_compress2".format(str(stage), str(c), mode)
                    weight_c2 = getattr(self, "weight_" + key)
                    for r in [0, 1, 2, 3]:
                        pred += torch.rot90(
                            self.InterpTorchBatch(weight_c1, weight_c2, mode, F.pad(torch.rot90(x_c, r, [
                                2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                        pred = self.round_func(pred)
            pred = pred / self.c_in
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))

        if self.c_in == 1:
            x = x.reshape((B, C, H, W))
        if phase == 'train':
            x = x / 255.0
        return x

class SPF_LUT_DFC(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, sigma, interval, compressed_dimensions, diagonal_width, sampling_interval, c_in = 1):
        super(SPF_LUT_DFC, self).__init__()
        self.interval = interval
        self.sigma = sigma
        self.c_in = c_in
        self.modes = modes
        self.stages = stages
        self.d = diagonal_width
        self.compression = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1
        # self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_L{}_d{}.npy'.format(L, d)))
        # self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        out_c_list = [2,2,2,1,-1,1]
        in_c_list = [1,1,1,1,-1,4]

        for s in range(6):
            stage = s+1
            if stage == 5:
                continue
            channels = in_c_list[s]
            for c in range(channels):
                for mode in modes:
                    lut_path = os.path.join(lut_folder, "{}_s{}c{}_{}_compress1.npy".
                                            format(lutName, str(stage), str(c), mode))
                    if not os.path.exists(lut_path):
                        continue
                    key = 's{}c{}_{}_compress1'.format(stage,c,mode)
                    lut_arr = np.load(lut_path).reshape((-1, out_c_list[s])).astype(np.float32) / 127.0
                    self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                    lut_path = os.path.join(lut_folder, "{}_s{}c{}_{}_compress2.npy".
                                            format(lutName, str(stage), str(c), mode))
                    if not os.path.exists(lut_path):
                        continue
                    key = 's{}c{}_{}_compress2'.format(stage, c, mode)
                    lut_arr = np.load(lut_path).reshape((-1, out_c_list[s])).astype(np.float32) / 127.0
                    self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))


        # channel stage5
        lut_path = os.path.join(lut_folder, "{}_s{}_channel.npy".format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch_compress1_xyzt(self, weight_c1,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17


        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]


        p0000 = weight_c1[k0000].reshape((-1, out_c))
        p0001 = weight_c1[k0001].reshape((-1, out_c))
        p0010 = weight_c1[k0010].reshape((-1, out_c))
        p0011 = weight_c1[k0011].reshape((-1, out_c))
        p0100 = weight_c1[k0100].reshape((-1, out_c))
        p0101 = weight_c1[k0101].reshape((-1, out_c))
        p0110 = weight_c1[k0110].reshape((-1, out_c))
        p0111 = weight_c1[k0111].reshape((-1, out_c))

        p1000 = weight_c1[k1000].reshape((-1, out_c))
        p1001 = weight_c1[k1001].reshape((-1, out_c))
        p1010 = weight_c1[k1010].reshape((-1, out_c))
        p1011 = weight_c1[k1011].reshape((-1, out_c))
        p1100 = weight_c1[k1100].reshape((-1, out_c))
        p1101 = weight_c1[k1101].reshape((-1, out_c))
        p1110 = weight_c1[k1110].reshape((-1, out_c))
        p1111 = weight_c1[k1111].reshape((-1, out_c))

        out = torch.zeros((img_a1.shape[0], out_c), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c))

        out = out / q
        return out, index_flag

    def InterpTorchBatch(self, weight_c1,weight_c2,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, out_c, mode, img_in, bd)
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval  
        L = 2 ** (8 - interval) + 1 

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        sz0,sz1,sz2,sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c))
            out_all = out_all.permute(0, 1, 4, 2, 3).reshape((sz0, sz1 * out_c, sz2, sz3))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c))


        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0,sz1,sz2,sz3, out_c))
        out_all = out_all.permute(0, 1, 4, 2, 3).reshape((sz0,sz1*out_c,sz2,sz3))
        return out_all

    def InterpTorchBatch_channel(self, weight, out_c, img_in):

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        # pytorch 1.5 dont support rounding_mode, use // equavilent
        # https://pytorch.org/docs/1.5.0/torch.html#torch.div
        img_a1, img_b1, img_c1, img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa, fb, fc, fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c))
        out = out.permute(0, 1, 4, 2, 3).reshape(
            (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2], img_a1.shape[3]))
        return out

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B * C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2,2,2,1]
        in_c = [1,1,1,1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(str(stage), 0, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(str(stage), 0, mode)
                weight_c2 = getattr(self, "weight_" + key)
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1,weight_c2,out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s]!=1:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                refine_list.append(x1)
                x = x2
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5 channel
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight,4,x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))
        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:,c:c+1,:,:]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1,weight_c2, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H, W))
        if phase == 'train':
            x = x / 255.0
        return x