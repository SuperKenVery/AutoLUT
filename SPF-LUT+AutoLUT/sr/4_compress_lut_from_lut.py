import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model as Model


def compress_from_lut_xy(L,d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    ref2index = np.zeros((L, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            if abs(x-y) <= d:
                index_x.append(x)
                index_y.append(y)
                ref2index[x,y-x] = cnt
                cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)

    index_compress = index_x * L + index_y
    return index_compress,ref2index

def compress_from_lut_xyz(L,d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    index_z = []
    ref2index = np.zeros((L, diag, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if abs(x-y) <= d and abs(x-z)<=d:
                    index_x.append(x)
                    index_y.append(y)
                    index_z.append(z)
                    ref2index[x,y-x,z-x] = cnt
                    cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)

    index_compress = index_x * L * L + index_y * L + index_z
    return index_compress,ref2index

def compress_from_lut_xyzt(L, d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    index_z = []
    index_t = []
    ref2index = np.zeros((L, diag, diag, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    if abs(x-y) <= d and abs(x-z)<=d and abs(x-t)<=d:
                        index_x.append(x)
                        index_y.append(y)
                        index_z.append(z)
                        index_t.append(t)
                        ref2index[x,y-x,z-x,t-x] = cnt
                        cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)
    index_t = np.array(index_t, dtype=np.int_)

    index_compress = index_x * L * L * L + index_y * L * L + index_z * L + index_t
    return index_compress,ref2index

def compress_from_lut_larger_interval(lut,L,interval):
    B, C = lut.shape
    lut = lut.reshape(L, L, L, L, C)

    if interval==5:
        k = 2
    elif interval==6:
        k = 4
    else:
        k = 8

    compressed_lut = lut[::k, ::k, ::k, ::k, :].reshape(-1, C)
    return compressed_lut

def compress_MuLUT_from_lut(opt):
    stages = opt.stages
    exp_dir = opt.expDir
    compression_type = opt.cd
    d = opt.dw
    interval = opt.si
    save_dir = os.path.join(exp_dir, '{}{}i{}'.format(compression_type, d, interval))
    modes = opt.modes
    L = 17

    if compression_type =='xy':
        index_compress, ref2index = compress_from_lut_xy(L, d)
    elif compression_type == 'xyz':
        index_compress, ref2index = compress_from_lut_xyz(L, d)
    else:
        index_compress,ref2index = compress_from_lut_xyzt(L,d)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)), ref2index)
    

    for s in range(stages):
        for mode in modes:
            lut_path = os.path.join(exp_dir, '{}_s{}_{}.npy'.format(opt.lutName, s + 1, mode))
            lut = np.load(lut_path)
            _,C = lut.shape

            if compression_type == 'xy':
                lut_c1 = lut.reshape((L*L,L,L,-1))[index_compress,:,:, :].reshape((-1,C))
            elif compression_type == 'xyz':
                lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
            else:
                lut_c1 = lut[index_compress,:]
            lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

            save_path = os.path.join(save_dir, '{}_s{}_{}_compress1.npy'.format(opt.lutName, s + 1, mode))
            np.save(save_path, lut_c1)
            print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

            save_path = os.path.join(save_dir, '{}_s{}_{}_compress2.npy'.format(opt.lutName, s + 1, mode))
            np.save(save_path, lut_c2)
            print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

def compress_SPFLUT_from_lut(opt):
    exp_dir = opt.expDir
    compression_type = opt.cd
    d = opt.dw
    interval = opt.si
    save_dir = os.path.join(exp_dir, '{}{}i{}'.format(compression_type, d, interval))
    modes = opt.modes
    L = 17

    if compression_type =='xy':
        index_compress, ref2index = compress_from_lut_xy(L, d)
    elif compression_type == 'xyz':
        index_compress, ref2index = compress_from_lut_xyz(L, d)
    else:
        index_compress,ref2index = compress_from_lut_xyzt(L, d)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)), ref2index)

    for s in range(6):
        if s+1==5:
            lut_path = os.path.join(exp_dir, '{}_s{}_channel.npy'.format(opt.lutName, s + 1))
            lut = np.load(lut_path)
            save_path = os.path.join(save_dir, '{}_s{}_channel.npy'.format(opt.lutName, s + 1))
            np.save(save_path, lut)
            print("Resulting LUT size: ", lut.shape, "Saved to", save_path)
            continue
        for mode in modes:
            if s+1==6:
                for c in range(4):
                    lut_path = os.path.join(exp_dir, '{}_s{}c{}_{}.npy'.format(opt.lutName, s + 1, c, mode))
                    lut = np.load(lut_path)
                    _,C = lut.shape
                    # print('weight_s{}c{}_{}.npy'.format(s + 1, c, mode),lut.shape)

                    if compression_type == 'xy':
                        lut_c1 = lut.reshape((L*L,L,L,-1))[index_compress,:,:, :].reshape((-1,C))
                    elif compression_type == 'xyz':
                        lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                    else:
                        lut_c1 = lut[index_compress,:]

                    lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                    save_path = os.path.join(save_dir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, s + 1, c, mode))
                    np.save(save_path, lut_c1)
                    print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                    save_path = os.path.join(save_dir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, s + 1, c, mode))
                    np.save(save_path, lut_c2)
                    print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)
            else:
                lut_path = os.path.join(exp_dir, '{}_s{}c0_{}.npy'.format(opt.lutName, s + 1, mode))
                lut = np.load(lut_path)
                _, C = lut.shape
                # print('weight_s{}c0_{}.npy'.format(s + 1, mode), lut.shape)

                if compression_type == 'xy':
                    lut_c1 = lut.reshape((L * L, L, L, -1))[index_compress, :, :, :].reshape((-1, C))
                elif compression_type == 'xyz':
                    lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                else:
                    lut_c1 = lut[index_compress, :]
                lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                save_path = os.path.join(save_dir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, s + 1, mode))
                np.save(save_path, lut_c1)
                print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                save_path = os.path.join(save_dir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, s + 1, mode))
                np.save(save_path, lut_c2)
                print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

if __name__ == "__main__":
    opt_inst = TestOptions()
    opt = opt_inst.parse()

    if opt.model == 'SPF_LUT_net':
        compress_SPFLUT_from_lut(opt)
    elif opt.model == 'BaseSRNets':
        compress_MuLUT_from_lut(opt)
    else:
        raise ValueError
    