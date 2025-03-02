import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, "../")  # run under the current directory
from common.network import *
from loguru import logger
from jaxtyping import jaxtyped, Float, Int64 as Long
from typeguard import typechecked as typechecker
from typing import Any, Self, Tuple, List, Dict

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}

def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def rotate_input(module, x, last_x, pad):
    pred = 0
    for r in [0,1,2,3]:
        rot_input=F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')
        rot_last_input=None if last_x is None else F.pad(torch.rot90(last_x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')
        rot_output=module(rot_input, rot_last_input)
        output=torch.rot90(rot_output, (4 - r) % 4, [2, 3])
        quantized=round_func(output * 127)
        pred+=quantized
    return pred

class SRNets(nn.Module):
    """ A LUT-convertable SR network with configurable stages and patterns. """

    def __init__(self, num_samplers, sample_size, nf=64, scale=4, stages=2, act='relu'):
        super(SRNets, self).__init__()

        self.stages=stages
        self.scale=scale
        self.nf=nf

        self.sample_size=sample_size
        self.num_samplers=num_samplers

        for s in range(stages):  # 2-stage
            if (s + 1) == stages:
                upscale = scale
            else:
                upscale = 1

            for i in range(self.num_samplers):
                self.add_module(
                    "s{}_{}".format(str(s), i),
                    SRNet(
                        self.sample_size,
                        nf=nf,
                        upscale=upscale,
                        residual = (s+1 > 1),
                        act=act,
                    )
                )
        # print_network(self)

    def forward(self,x,phase='train'):
        last_x=None
        for stage in range(self.stages):
            pred=0

            for i in range(self.num_samplers):
                module=getattr(self, f"s{stage}_{i}")
                pad=self.sample_size-1
                pred += rotate_input(module,x,last_x,pad)

            last_x=x

            if stage+1 == self.stages:
                # Last stage
                avg_factor = self.num_samplers
                x = round_func(pred / avg_factor)
                if phase=='train':
                    x = x/255
            else:
                avg_factor, bias, norm = self.num_samplers*4, 127, 255
                x = round_func(torch.clamp(pred/avg_factor+bias,0,255)) / norm

        return x



class AutoLUT(nn.Module):
    """
    Inference AutoLUT using exported look-up table
    Use pytorch to finetune the LUT
     """

    def __init__(self, lut_folder, stages, samplers, sample_size, upscale=4, interval=4):
        super(AutoLUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.stages: int = stages
        self.samplers: int = samplers
        self.sample_size: int = sample_size

        for stage in range(stages):
            scale = upscale if stage == stages-1 else 1
            residual = (stage > 0)
            for sampler in range(self.samplers):
                lut_path = os.path.join(lut_folder,
                                        "LUT_x{}_{}bit_int8_s{}_{}.npy".format(upscale, interval, stage, sampler))
                key = "s{}_{}".format(str(stage), sampler)
                lut_arr = np.load(lut_path).reshape(-1, scale * scale).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.tensor(lut_arr)))

                # Auto sampler and residual weights don't need to be fine-tuned, so we don't register parameter

                # Load auto smapler
                sampler_path=os.path.join(lut_folder, f"LUT_x{upscale}_{interval}bit_int8_s{stage}_{sampler}_sampler.npy")
                sampler_key = key+"_sampler"
                sampler_weights=torch.Tensor(np.load(sampler_path))
                out_channel, in_channel, kernel_w, kernel_h = sampler_weights.shape
                assert kernel_h==kernel_w==self.sample_size
                sampler_layer=AutoSample(kernel_h)
                sampler_layer.sampler.weight=nn.Parameter(sampler_weights)
                sampler_layer.sampler.bias=nn.Parameter(torch.zeros(sampler_layer.sampler.bias.shape))
                self.add_module(sampler_key, sampler_layer)

                # Load residual
                residual_key = key+"_residual"
                if residual:
                    residual_path=os.path.join(lut_folder, f"LUT_x{upscale}_{interval}bit_int8_s{stage}_{sampler}_residual.npy")
                    residual_weights=torch.Tensor(np.load(residual_path))
                    residual_weights=nn.Parameter(residual_weights)
                else:
                    residual_weights=None
                self.register_parameter(residual_key, residual_weights)

    def rotate_input(self, weight, scale, sampler, x, pad, res_w, prev_x):
        pred=0
        for r in [0,1,2,3]:
            pred += torch.rot90(
                self.InterpTorchBatch(
                    weight,
                    res_w,
                    sampler,
                    scale,
                    F.pad(
                        torch.rot90(x, r, [2, 3]),
                        (0, pad, 0, pad),
                        mode='replicate'
                    ),
                    F.pad(
                        torch.rot90(prev_x, r, [2, 3]),
                        (0, pad, 0, pad),
                        mode='replicate'
                    ) if prev_x!=None else None,
                    pad
                ),
                (4 - r) % 4,
                [2, 3]
            )
            pred = self.round_func(pred)
        return pred

    def forward(self, x, phase='train'):
        x = x * 255.0
        prev_x = None
        numSamplers, stages = self.samplers, self.stages
        for stage in range(stages):
            pred = 0
            if stage == stages - 1:
                avg_factor, bias, norm = numSamplers, 0, 1
                scale = self.upscale
            else:
                avg_factor, bias, norm = numSamplers * 4, 127, 255.0
                scale = 1
            for sampler in range(numSamplers):
                pad = self.sample_size-1
                key = "s{}_{}".format(str(stage), sampler)
                weight = getattr(self, "weight_" + key)
                sampler = getattr(self, key+"_sampler")
                residual_weights = getattr(self, key+"_residual")
                pred+=self.rotate_input(weight, scale, sampler, x, pad, residual_weights, prev_x)
            prev_x = x
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))

        if phase=='train':
            x = x / 255.0
        return x

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def sample(self, sampler: AutoSample, img: torch.Tensor) -> torch.Tensor:
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

    def fix_weight(self,weight):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)
        return weight

    def InterpTorchBatch(self, weight: torch.Tensor, res_w: torch.Tensor, sampler: AutoSample,
                         upscale, img_in, prev_img, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = self.fix_weight(weight)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

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

        p0000 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0001 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0010 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0011 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0100 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0101 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0110 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p0111 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))

        p1000 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1001 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1010 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1011 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1100 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1101 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1110 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        p1111 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()
                       ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], upscale, upscale), dtype=weight.dtype).to(device=weight.device)
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
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
        out = out / q
        return out

from common.lut import LUT_Symmetric

class SymmetricLUTNet(AutoLUT):
    def __init__(self, samplers, sample_size, nf, scale, stages, act):
        interval=4
        upscale=scale
        nn.Module.__init__(self)
        self.interval = interval
        self.upscale = upscale
        self.stages: int = stages
        self.samplers: int = samplers
        self.sample_size: int = sample_size

        for stage in range(stages):
            scale = upscale if stage == stages-1 else 1
            residual = (stage > 0)
            for sampler in range(self.samplers):
                key = "s{}_{}".format(str(stage), sampler)

                self.add_module('weight_'+key, LUT_Symmetric(
                    interval,
                    upscale if stage == stages-1 else 1,
                ))

                # Load auto smapler
                sampler_key = key+"_sampler"
                sampler_layer=AutoSample(self.sample_size)
                self.add_module(sampler_key, sampler_layer)

                # Load residual
                residual_key = key+"_residual"
                if residual:
                    residual_weights=torch.rand(2,2)
                    residual_weights=nn.Parameter(residual_weights)
                else:
                    residual_weights=None
                self.register_parameter(residual_key, residual_weights)

    def fix_weight(self, ls: LUT_Symmetric) -> LUT_Symmetric:
        return ls

if __name__=='__main__':
    print("main")
    sl=SymmetricLUTNet(3,3,1,4,2,'').cuda()
    x=torch.rand(10,1,128,128).cuda()
    opt=torch.optim.Adam(sl.parameters(), lr=0.001)

    print("forward")
    y=sl(x)
    loss=y.sum()
    loss.backward()
    # opt.step()
    grad=sl.weight_s0_0.w0.grad
    print(grad)


