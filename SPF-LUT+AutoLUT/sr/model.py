import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

sys.path.insert(0, "../")  # run under the current directory
from common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def identity(input):
    return input


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, sample_size, scale=None, output_quant=False, modes=['s', 'd', 'y'], nf=64):
        super(ConvBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.module_dict = dict()
        self.upscale = scale
        self.output_quant = output_quant
        self.sample_size=sample_size

        scale_factor = 1 if scale is None else scale ** 2
        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = MuLUTConv('{}x{}'.format(mode.upper(), 'N'),
                                                                                    nf=nf, sample_size=sample_size, out_c=out_c * scale_factor,
                                                                                    stride=1)
        self.module_dict = nn.ModuleDict(self.module_dict)
        if scale is None:
            self.pixel_shuffle = identity
        else:
            self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x, prev_x):
        modes = self.modes

        x_out = 0
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            prevx_c = prev_x[:, c:c+1, :, :] if prev_x!=None else None
            pred = 0
            for mode in modes:
                # pad = mode_pad_dict[mode]
                pad = self.sample_size-1
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    y = round_func(
                        torch.tanh(
                            torch.rot90(
                                self.pixel_shuffle(
                                    sub_module(
                                        F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate'),
                                        F.pad(torch.rot90(prevx_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate') if prevx_c!=None else None
                                    )
                                ),
                                (4 - r) % 4, [2, 3]
                            )
                        ) * 127
                    )

                    assert isinstance(pred, torch.Tensor)==False or pred.shape==y.shape, f"Unexpected shape: pred={pred.shape}, y={y.shape}"
                    pred += y

            x_out += pred
        if self.output_quant:
            avg_factor = len(modes) * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            x = x_out / self.in_c

        return x


class SPF_LUT_net(nn.Module):
    def __init__(self, sample_size, nf=32, scale=4, modes=['s', 'd', 'y'], stages=2):
        super(SPF_LUT_net, self).__init__()
        self.upscale = scale
        self.modes = modes

        self.convblock1 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf, sample_size=sample_size)
        self.convblock2 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf, sample_size=sample_size)
        self.convblock3 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf, sample_size=sample_size)
        self.convblock4 = ConvBlock(1, 1, scale=None, output_quant=False, modes=modes, nf=nf, sample_size=sample_size)
        self.ChannelConv = MuLUTcUnit(in_c=4, out_c=4, mode='1x1', nf=nf)
        self.upblock = ConvBlock(4, 1, scale=scale, output_quant=False, modes=modes, nf=nf, sample_size=sample_size)


    def forward(self, x, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        # block1
        x1 = x
        x = self.convblock1(x, None)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block2
        x2 = x
        x = self.convblock2(x, x1)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block3
        x3 = x
        x = self.convblock3(x, x2)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block4
        x4 = x
        x = self.convblock4(x, x3)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        refine_list.append(x)

        # concat
        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        # upblock
        x = self.upblock(x, torch.cat([x1, x2, x3, x4], dim=1))
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H, self.upscale * W))

        return x

# lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, phase = 'train'
class SPF_LUT_DFC(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, sample_size, phase = 'train'):
        super(SPF_LUT_DFC, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.d = diagonal_width
        L = 2 ** (8 - interval) + 1
        self.compression_type = compressed_dimensions
        # self.diagonal_width = diagonal_width
        self.sampling_interval = sampling_interval
        self.sample_size=sample_size

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            ref2index = torch.Tensor(ref2index).type(torch.int64)
            self.register_buffer('ref2index', ref2index)
        else:
            self.ref2index = None

        def load_sampler_and_residual(stage, mode, channel=0):
            # Load AutoSampler
            sampler_path=os.path.join(lut_folder, 'sampler_s{}c{}_{}.npy'.format(stage, channel, mode))
            sampler_weights=np.load(sampler_path)
            sampler_weights=torch.tensor(sampler_weights)
            sampler=AutoSample(sample_size)
            sampler.sampler.weight=nn.Parameter(sampler_weights)
            self.add_module('sampler_s{}c{}_{}'.format(stage, channel, mode), sampler)

            # Load residual
            res_path=os.path.join(lut_folder, 'residual_s{}c{}_{}.npy'.format(stage, channel, mode))
            res_weights=torch.tensor(np.load(res_path))
            self.register_parameter('residual_s{}c{}_{}'.format(stage, channel, mode), nn.Parameter(res_weights))


        for mode in modes:
            # conv1
            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress1".format(1, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress2".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            load_sampler_and_residual(1, mode)

            # conv2
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress1".format(2, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress2".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            load_sampler_and_residual(2, mode)

            # conv3
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress1".format(3, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress2".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            load_sampler_and_residual(3, mode)

            # conv4
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress1".format(4, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress2".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            load_sampler_and_residual(4, mode)

            for c in range(4):
                # conv6
                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c{}_{}_compress1.npy'.format(lutName, upscale, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                if compressed_dimensions=='xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c{}_{}_compress2.npy'.format(lutName, upscale, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                load_sampler_and_residual(6, mode, channel=c)

        # conv5
        if phase == 'train':
            lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}_channel.npy'.format(lutName, upscale, 5))
        else:
            lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
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

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_abcd, bd):
        img_a, img_b, img_c, img_d = img_abcd
        _, _, h, w = img_a.shape
        # h -= bd
        # w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_x = img_a
        img_y = img_b
        img_z = img_c
        img_t = img_d
        index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
        index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
        index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
        index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt
        if not index_flag.any():
            out = torch.zeros((0,1), dtype=weight_c1.dtype).to(device=weight_c1.device)
            return out, index_flag

        # Extract MSBs
        img_a1 = torch.floor_divide(img_a, q).type(torch.int64)
        img_b1 = torch.floor_divide(img_b, q).type(torch.int64)
        img_c1 = torch.floor_divide(img_c, q).type(torch.int64)
        img_d1 = torch.floor_divide(img_d, q).type(torch.int64)

        # Extract LSBs
        fa = img_a % q
        fb = img_b % q
        fc = img_c % q
        fd = img_d % q

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

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
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
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    @staticmethod
    def unfold(K, P, x):
        """
        Do the convolution sampling
        """
        if x is None: return x, None
        B, C, H, W = x.shape
        x = F.unfold(x, K)  # B,C*K*K,L
        x = x.view(B, C, K * K, (H - P) * (W - P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - P) * (W - P),
                      K, K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,1,K,K

        return x, (B, C, H, W)

    @staticmethod
    def put_back(P, x, ori_shape):
        B, C, H, W=ori_shape
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        x = F.fold(x, ((H - P), (W - P)),
                   1, stride=1)
        return x

    def sample(self, sampler: AutoSample, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unfolded, shape = self.unfold(sampler.input_shape, sampler.input_shape-1, img)
        assert unfolded.shape[-2:]==(sampler.input_shape, sampler.input_shape), f"Unexpected shape after unfold: {unfolded.shape}"
        # unfolded: B*C*L,1,K,K
        sampled = sampler(unfolded)
        # sampled: B*C*L,1,2,2
        assert sampled.shape[:-2]==unfolded.shape[:-2] and sampled.shape[-2:]==(2,2)
        _a=sampled[:,:,0,0].unsqueeze(-1).unsqueeze(-1)
        _b=sampled[:,:,0,1].unsqueeze(-1).unsqueeze(-1)
        _c=sampled[:,:,1,0].unsqueeze(-1).unsqueeze(-1)
        _d=sampled[:,:,1,1].unsqueeze(-1).unsqueeze(-1)

        assert _a.shape==_b.shape==_c.shape==_d.shape
        a=self.put_back(sampler.input_shape-1, _a, shape)
        b=self.put_back(sampler.input_shape-1, _b, shape)
        c=self.put_back(sampler.input_shape-1, _c, shape)
        d=self.put_back(sampler.input_shape-1, _d, shape)

        return a,b,c,d

    @staticmethod
    def inrange(lo, hi, *xs):
        for x in xs:
            if not ( (lo<=x).all() and (x<=hi).all() ):
                return False
        return True

    def InterpTorchBatch(self, weight_c1, weight_c2, sampler: AutoSample, res_w, upscale,out_c, mode, img_in, prev_img, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        #Auto Sample
        a,b,c,d=self.sample(sampler, img_in)

        # Residual
        if prev_img != None:
            res_w = torch.clamp(res_w, 0, 1)
            oa,ob,oc,od=a,b,c,d
            pa,pb,pc,pd=self.sample(sampler, prev_img)

            a = res_w[0][0]*pa + (1-res_w[0][0])*oa
            b = res_w[0][1]*pb + (1-res_w[0][1])*ob
            c = res_w[1][0]*pc + (1-res_w[1][0])*oc
            d = res_w[1][1]*pd + (1-res_w[1][1])*od

            #debug
            _a=pa.reshape([-1, 1, 1, 1])
            _b=pb.reshape([-1, 1, 1, 1])
            _c=pc.reshape([-1, 1, 1, 1])
            _d=pd.reshape([-1, 1, 1, 1])
            upper = torch.cat([_a,_b], dim=3)
            lower = torch.cat([_c,_d], dim=3)
            sampled = torch.cat([upper, lower], dim=2)
            # print(f"LUT sampled prevx {sampled}")
            # print(f"LUT got prevx {prev_img}")

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        img_a1 = torch.floor_divide(a, q).type(torch.int64)
        img_b1 = torch.floor_divide(b, q).type(torch.int64)
        img_c1 = torch.floor_divide(c, q).type(torch.int64)
        img_d1 = torch.floor_divide(d, q).type(torch.int64)

        fa = a % q
        fb = b % q
        fc = c % q
        fd = d % q

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        assert self.compression_type=='xyzt'
        out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(
            weight_c1, upscale,out_c, mode,
            (a,b,c,d),
            bd
        )

        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c * upscale * upscale), dtype=weight_c2.dtype).to(
            device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all
        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        assert self.inrange(0, 255/L, img_a1, img_b1, img_c1, img_d1, img_a2, img_b2, img_c2, img_d2)
        key=img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()
        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        # out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
        #                    img_a1.shape[3], out_c*upscale*upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        # sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        # out_all = out.reshape(sz, -1)
        # out = out_all[~index_flag]

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
        out_all = out_all.reshape((sz0,sz1,sz2,sz3, out_c,upscale, upscale))
        out_all = out_all.permute(0,1,4,2,5,3,6).reshape(
            (sz0, sz1*out_c, sz2 * upscale, sz3 * upscale))
        return out_all

    def InterpTorchBatch_channel(self, weight,out_c, img_in):

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        # pytorch 1.5 dont support rounding_mode, use // equavilent
        # https://pytorch.org/docs/1.5.0/torch.html#torch.div
        img_a1,img_b1,img_c1,img_d1 = torch.chunk(
            torch.floor_divide(img_in, q).type(torch.int64),
            4,1
        )

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in%q,4,1)

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
        out = out.permute(0,1,4,2,3).reshape(
            (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))
        return out

    def forward(self, x, phase='train'):
        B,C,H,W = x.shape
        x = x.reshape((B*C,1,H,W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2,2,2,1]
        refine_list = []
        xs = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = self.sample_size-1
                key = "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale =1
                sampler_key='sampler_s{}c{}_{}'.format(stage, 0, mode)
                sampler=getattr(self, sampler_key)
                res_w_key='residual_s{}c{}_{}'.format(stage, 0, mode)
                res_w=getattr(self, res_w_key)
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(
                            weight_c1, weight_c2, sampler, res_w,
                            scale,out_c_list[s], mode,
                            F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate'),
                            F.pad(torch.rot90(xs[-1], r, [2, 3]), (0, pad, 0, pad), mode='replicate') if s>0 else None,
                            pad
                        ), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            xs.append(x)
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s]==2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)

        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight,4,x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:,c:c+1,:,:]
            prevx_c=xs[c]
            for mode in self.modes:
                pad = self.sample_size-1
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                sampler_key='sampler_s{}c{}_{}'.format(6, c, mode)
                sampler=getattr(self, sampler_key)
                res_w_key='residual_s{}c{}_{}'.format(6, c, mode)
                res_w=getattr(self, res_w_key)
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(
                            weight_c1, weight_c2, sampler, res_w,
                            scale, 1, mode,
                            F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),mode='replicate'),
                            F.pad(torch.rot90(prevx_c, r, [2, 3]), (0, pad, 0, pad),mode='replicate'),
                            pad
                            ),
                        (4 - r) % 4,[2, 3]
                        )
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B,C,H*self.upscale,W*self.upscale))
        if phase == 'train':
            x = x / 255.0
        return x
