import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
from common.network import MuLUTUnit, SRNet, AutoSample
from model import SRNets
from loguru import logger
import model


def get_input_tensor(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    # base[-1] -= 1
    L = base.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 2, 2).float() / 255.0
    return input_tensor


def check_weights(weights: np.array, input_size: int):
    """
        Check AutoSampler's weights after softmax

        Asserts that on the last 2 dims they sum to 1

        input_size: The input size (length of one edge) of the sampler
    """
    sums=np.sum(weights, axis=(2,3))
    assert all(np.abs(sums-np.ones(sums.shape))<1e-3), f"Sum mismatch: {weights}, sums={sums}"
    # print(f"sums are {sums}")

if __name__ == "__main__":
    opt_inst = TestOptions()
    opt = opt_inst.parse()

    # load model
    opt = TestOptions().parse()

    model = getattr(model, opt.model)

    model_G: SRNets = model(nf=opt.nf, scale=opt.scale, stages=opt.stages, num_samplers=opt.numSamplers, sample_size=opt.sampleSize, act=opt.activateFunction).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    for stage in range(opt.stages):
        for sampler in range(opt.numSamplers):
            input_tensor = get_input_tensor(opt)

            # Split input to not over GPU memory
            B = input_tensor.size(0) // 100
            outputs = []

            # Extract the corresponding part from network
            name=f"s{stage}_{sampler}"
            srnet: SRNet = getattr(model_G, name)
            auto_sampler: AutoSample = srnet.sampler
            # weight shape: [out_channels, in_channels, kernel_size, kernel_size]
            raw_sw = auto_sampler.sampler.weight
            sampler_weights = raw_sw.cpu().detach().numpy()
            # check_weights(sampler_weights, auto_sampler.input_shape)
            sampler_path = os.path.join(opt.expDir, f"LUT_x{opt.scale}_{opt.interval}bit_int8_s{stage}_{sampler}_sampler.npy")
            np.save(sampler_path, sampler_weights)


            mulutunit: MuLUTUnit = srnet.model

            if mulutunit.has_residual:
                residual_wegiths = mulutunit.residual.weights.cpu().detach().numpy()
                residual_path = os.path.join(opt.expDir, f"LUT_x{opt.scale}_{opt.interval}bit_int8_s{stage}_{sampler}_residual.npy")
                np.save(residual_path, residual_wegiths)

            # Force inference without residual
            mulutunit.has_residual=False

            # Extract input-output pairs
            with torch.no_grad():
                model_G.eval()
                for b in range(100+1):
                    batch_input = input_tensor[b * B:(b + 1) * B]

                    batch_output = mulutunit(batch_input)
                    assert (-1<=batch_output).all() and (batch_output<=1).all(), f"Output out of bound: {batch_output}"

                    results = torch.round(torch.clamp(batch_output, -1, 1)
                                          * 127).cpu().data.numpy().astype(np.int8)
                    outputs += [results]

            results = np.concatenate(outputs, 0)

            lut_path = os.path.join(opt.expDir,
                                    "LUT_x{}_{}bit_int8_s{}_{}.npy".format(opt.scale, opt.interval, stage, sampler))
            np.save(lut_path, results)

            print(f"Conv layers ({results.shape}) -> {lut_path}, sampler ({sampler_weights.shape}) -> {sampler_path}")
