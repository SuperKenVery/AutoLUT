import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
from common.network import MuLUTConv, MuLUTConvUnit
import model as Model


def get_input_tensor(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
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

def get_mode_input_tensor(input_tensor, mode):
    if mode == "d":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    elif mode == "y":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    return input_tensor

def compress_lut(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    input_tensor = input_tensor.reshape(L * L, L, L, 1, 2, 2)
    index_i = torch.zeros((N,)).type(torch.int64)
    index_j = torch.zeros((N,)).type(torch.int64)
    cnt = 0
    ref2index = np.zeros((L, diag), dtype=np.int_) - 1
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= d:
                index_i[cnt] = i
                index_j[cnt] = j
                ref2index[i, j - i] = cnt
                cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    index_compress = index_i * L + index_j
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def compress_lut_xyz(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L, L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if abs(x - y) <= d and abs(x - z) <= d:
                    ref_x.append(x)
                    ref_y.append(y)
                    ref_z.append(z)
                    ref2index[x, y - x, z - x] = cnt
                    cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)

    index_compress = ref_x * L * L + ref_y * L + ref_z
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut_xyzt(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L * L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    ref_t = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    if abs(x - y) <= d and abs(x - z) <= d and abs(x - t) <= d:
                        ref_x.append(x)
                        ref_y.append(y)
                        ref_z.append(z)
                        ref_t.append(t)
                        ref2index[x, y - x, z - x, t - x] = cnt
                        cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)
    ref_t = torch.Tensor(ref_t).type(torch.int64)

    index_compress = ref_x * L * L * L + ref_y * L * L + ref_z * L + ref_t
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def compress_lut_larger_interval(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    input_tensor = input_tensor.reshape(L, L, L, L, 1, 2, 2)

    if opt.si==5:
        k = 2
    elif opt.si==6:
        k = 4
    elif opt.si==7:
        k = 8
    else:
        raise ValueError

    compressed_input_tensor = input_tensor[::k, ::k, ::k, ::k, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def save_lut(input_tensor, lut_path, s, mode, model_G):
    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100
    outputs = []

    # Extract input-output pairs
    with torch.no_grad():
        model_G.eval()
        for b in range(100):
            if b == 99:
                batch_input = input_tensor[b * B:]
            else:
                batch_input = input_tensor[b * B:(b + 1) * B]

            batch_output = getattr(model_G, "s{}_{}".format(str(s + 1), mode))(batch_input)

            results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
            outputs += [results]

    results = np.concatenate(outputs, 0)
    results = results.reshape(input_tensor.size(0), -1)
    np.save(lut_path, results)
    print("Resulting LUT size: ", results.shape, "Saved to", lut_path)


def compress_SPFLUT(opt):
    def save_SPFLUT_DFC(x, lut_path, module):
        # Split input to not over GPU memory
        B = x.size(0) // 100
        outputs = []

        if isinstance(module, MuLUTConv):
            module=module.model

        # Extract input-output pairs
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]

                batch_output = module(batch_input, prev_x=None)

                results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def save_sampler_and_res(module: MuLUTConv, stage, mode, channel=0):
        # Save sampler
        sampler=module.sampler
        weights=sampler.sampler.weight.detach().cpu().numpy()
        np.save(os.path.join(opt.expDir, 'sampler_s{}c{}_{}.npy'.format(stage, channel, mode)), weights)

        # Save residual
        convunit: MuLUTConvUnit=module.model
        res=convunit.residual.weights.data.cpu().numpy()
        np.save(os.path.join(opt.expDir, 'residual_s{}c{}_{}.npy'.format(stage, channel, mode)), res)

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, 'SPF_LUT_net')

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages, sample_size=opt.sample_size).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm, strict=True)

    input_tensor = get_input_tensor(opt)
    for mode in modes:
        if opt.cd == 'xyzt':
            input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
        elif opt.cd == 'xyz':
            input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
        elif opt.cd == 'xy':
            input_tensor_c1 = compress_lut(opt, input_tensor)
        else:
            raise ValueError
        input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        save_SPFLUT_DFC(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        save_SPFLUT_DFC(input_tensor_c2, lut_path, module)
        save_sampler_and_res(module, 1, mode)

        # conv2
        module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        save_SPFLUT_DFC(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        save_SPFLUT_DFC(input_tensor_c2, lut_path, module)
        save_sampler_and_res(module, 2, mode)

        # conv3
        module = model_G.convblock3.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 3, mode))
        save_SPFLUT_DFC(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 3, mode))
        save_SPFLUT_DFC(input_tensor_c2, lut_path, module)
        save_sampler_and_res(module, 3, mode)

        # conv4
        module = model_G.convblock4.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        save_SPFLUT_DFC(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        save_SPFLUT_DFC(input_tensor_c2, lut_path, module)
        save_sampler_and_res(module, 4, mode)

        # conv6
        for c in range(4):
            module = model_G.upblock.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6,c, mode))
            save_SPFLUT_DFC(input_tensor_c1, lut_path, module)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6,c, mode))
            save_SPFLUT_DFC(input_tensor_c2, lut_path, module)
            save_sampler_and_res(module, 6, mode, c)

    # conv5
    input_tensor = input_tensor.reshape((-1,4,1,1))
    module = model_G.ChannelConv
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_SPFLUT_DFC(input_tensor, lut_path, module)


def compress_MuLUT(opt):
    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, 'BaseSRNets')

    model_G = model(nf=opt.nf, modes=modes, stages=stages, scale=opt.scale).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm, strict=True)

    for s in range(stages):
        stage = s + 1

        for mode in modes:
            input_tensor = get_input_tensor(opt)
            if opt.cd == 'xyzt':
                input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
            elif opt.cd == 'xyz':
                input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
            elif opt.cd == 'xy':
                input_tensor_c1 = compress_lut(opt, input_tensor)
            else:
                raise ValueError

            input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

            if mode != 's':
                input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)
                input_tensor_c2 = get_mode_input_tensor(input_tensor_c2, mode)

            lut_path = os.path.join(opt.expDir, '{}_s{}_{}_compress1.npy'.format(opt.lutName, str(stage), mode))
            save_lut(input_tensor_c1, lut_path, s, mode, model_G)

            lut_path = os.path.join(opt.expDir, '{}_s{}_{}_compress2.npy'.format(opt.lutName, str(stage), mode))
            save_lut(input_tensor_c2, lut_path, s, mode, model_G)

if __name__ == "__main__":
    opt_inst = TestOptions()
    opt = opt_inst.parse()

    if opt.model == 'SPF_LUT_net':
        compress_SPFLUT(opt)
    elif opt.model == 'BaseSRNets':
        compress_MuLUT(opt)
    else:
        raise ValueError
