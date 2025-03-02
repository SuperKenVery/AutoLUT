import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
from common.utils import PSNR, cal_ssim, modcrop, _rgb2ycbcr

# compression_type = 'xyzt'

def InterpTorchBatch_compress_xy(weight, img_in, h, w, interval, rot, d, upscale=4, out_c=1, mode='s',ref2index=None):
    q = 2 ** interval  # 16
    L = 2 ** (8 - interval) + 1  # 17

    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    if mode == "s":
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 1:1 + w]
        index_flag = (np.abs(img_x - img_y) <= d * q)

        # Extract MSBs
        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

    elif mode == 'd':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 2:2 + w]
        index_flag = (np.abs(img_x - img_y) <= d * q)

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, :, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

    elif mode == 'y':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 1:1 + h, 1:1 + w]
        index_flag = (np.abs(img_x - img_y) <= d * q)

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a1 = img_a1[index_flag].flatten().astype(np.int_)
    img_b1 = img_b1[index_flag].flatten().astype(np.int_)
    img_c1 = img_c1[index_flag].flatten().astype(np.int_)
    img_d1 = img_d1[index_flag].flatten().astype(np.int_)

    fa = fa[index_flag].flatten()
    fb = fb[index_flag].flatten()
    fc = fc[index_flag].flatten()
    fd = fd[index_flag].flatten()

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    k00 = ref2index[img_a1, img_b1 - img_a1]
    k01 = ref2index[img_a1, img_b2 - img_a1]
    k10 = ref2index[img_a2, img_b1 - img_a2]
    k11 = ref2index[img_a2, img_b2 - img_a2]

    p0000 = weight[k00,img_c1, img_d1].reshape((-1, out_c,upscale,upscale))
    p0001 = weight[k00,img_c1, img_d2].reshape((-1, out_c,upscale,upscale))
    p0010 = weight[k00,img_c2, img_d1].reshape((-1, out_c,upscale,upscale))
    p0011 = weight[k00,img_c2, img_d2].reshape((-1, out_c,upscale,upscale))
    p0100 = weight[k01,img_c1, img_d1].reshape((-1, out_c,upscale,upscale))
    p0101 = weight[k01,img_c1, img_d2].reshape((-1, out_c,upscale,upscale))
    p0110 = weight[k01,img_c2, img_d1].reshape((-1, out_c,upscale,upscale))
    p0111 = weight[k01,img_c2, img_d2].reshape((-1, out_c,upscale,upscale))

    p1000 = weight[k10,img_c1, img_d1].reshape((-1, out_c,upscale,upscale))
    p1001 = weight[k10,img_c1, img_d2].reshape((-1, out_c,upscale,upscale))
    p1010 = weight[k10,img_c2, img_d1].reshape((-1, out_c,upscale,upscale))
    p1011 = weight[k10,img_c2, img_d2].reshape((-1, out_c,upscale,upscale))
    p1100 = weight[k11,img_c1, img_d1].reshape((-1, out_c,upscale,upscale))
    p1101 = weight[k11,img_c1, img_d2].reshape((-1, out_c,upscale,upscale))
    p1110 = weight[k11,img_c2, img_d1].reshape((-1, out_c,upscale,upscale))
    p1111 = weight[k11,img_c2, img_d2].reshape((-1, out_c,upscale,upscale))

    # Output image holder
    out = np.zeros((img_a1.shape[0],out_c, upscale, upscale))
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

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    # print(p0000[i].shape,fa[i].shape,i.shape,out_c)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    out = out / q
    return out,index_flag

def InterpTorchBatch_compress_xyz(weight, img_in, h, w, interval, rot, d, upscale=4, out_c=1, mode='s',ref2index=None):
    q = 2 ** interval  # 16
    L = 2 ** (8 - interval) + 1  # 17

    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    if mode == "s":
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 1:1 + w]
        img_z = img_in[:, :, 1:1 + h, 0:0 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz)

        # Extract MSBs
        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

    elif mode == 'd':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 2:2 + w]
        img_z = img_in[:, :, 2:2 + h, 0:0 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz)

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, :, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

    elif mode == 'y':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 1:1 + h, 1:1 + w]
        img_z = img_in[:, :, 1:1 + h, 2:2 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz)

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))

    img_a1 = img_a1[index_flag].flatten().astype(np.int_)
    img_b1 = img_b1[index_flag].flatten().astype(np.int_)
    img_c1 = img_c1[index_flag].flatten().astype(np.int_)
    img_d1 = img_d1[index_flag].flatten().astype(np.int_)

    fa = fa[index_flag].flatten()
    fb = fb[index_flag].flatten()
    fc = fc[index_flag].flatten()
    fd = fd[index_flag].flatten()

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    k000 = ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
    k001 = ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
    k010 = ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
    k011 = ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

    k100 = ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
    k101 = ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
    k110 = ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
    k111 = ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

    p0000 = weight[k000, img_d1].reshape((-1, out_c,upscale,upscale))
    p0001 = weight[k000, img_d2].reshape((-1, out_c,upscale,upscale))
    p0010 = weight[k001, img_d1].reshape((-1, out_c,upscale,upscale))
    p0011 = weight[k001, img_d2].reshape((-1, out_c,upscale,upscale))
    p0100 = weight[k010, img_d1].reshape((-1, out_c,upscale,upscale))
    p0101 = weight[k010, img_d2].reshape((-1, out_c,upscale,upscale))
    p0110 = weight[k011, img_d1].reshape((-1, out_c,upscale,upscale))
    p0111 = weight[k011, img_d2].reshape((-1, out_c,upscale,upscale))

    p1000 = weight[k100, img_d1].reshape((-1, out_c,upscale,upscale))
    p1001 = weight[k100, img_d2].reshape((-1, out_c,upscale,upscale))
    p1010 = weight[k101, img_d1].reshape((-1, out_c,upscale,upscale))
    p1011 = weight[k101, img_d2].reshape((-1, out_c,upscale,upscale))
    p1100 = weight[k110, img_d1].reshape((-1, out_c,upscale,upscale))
    p1101 = weight[k110, img_d2].reshape((-1, out_c,upscale,upscale))
    p1110 = weight[k111, img_d1].reshape((-1, out_c,upscale,upscale))
    p1111 = weight[k111, img_d2].reshape((-1, out_c,upscale,upscale))

    # Output image holder
    out = np.zeros((img_a1.shape[0],out_c, upscale, upscale))
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

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    # print(p0000[i].shape,fa[i].shape,i.shape,out_c)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    out = out / q
    return out,index_flag

def InterpTorchBatch_compress_xyzt(weight, img_in, h, w, interval, rot, d, upscale=4, out_c=1, mode='s',ref2index=None):
    q = 2 ** interval  # 16
    L = 2 ** (8 - interval) + 1  # 17

    B,channels = weight.shape
    weight = weight.reshape((-1,channels))

    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    if mode == "s":
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 1:1 + w]
        img_z = img_in[:, :, 1:1 + h, 0:0 + w]
        img_t = img_in[:, :, 1:1 + h, 1:1 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag_xt = (np.abs(img_x - img_t) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

        # Extract MSBs
        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

    elif mode == 'd':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 0:0 + h, 2:2 + w]
        img_z = img_in[:, :, 2:2 + h, 0:0 + w]
        img_t = img_in[:, :, 2:2 + h, 2:2 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag_xt = (np.abs(img_x - img_t) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, :, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

    elif mode == 'y':
        img_x = img_in[:, :, 0:0 + h, 0:0 + w]
        img_y = img_in[:, :, 1:1 + h, 1:1 + w]
        img_z = img_in[:, :, 1:1 + h, 2:2 + w]
        img_t = img_in[:, :, 2:2 + h, 1:1 + w]
        index_flag_xy = (np.abs(img_x - img_y) <= d * q)
        index_flag_xz = (np.abs(img_x - img_z) <= d * q)
        index_flag_xt = (np.abs(img_x - img_t) <= d * q)
        index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

        img_a1 = img_in[:, :, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, :, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, :, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, :, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    
    img_a1 = img_a1[index_flag].flatten().astype(np.int_)
    img_b1 = img_b1[index_flag].flatten().astype(np.int_)
    img_c1 = img_c1[index_flag].flatten().astype(np.int_)
    img_d1 = img_d1[index_flag].flatten().astype(np.int_)

    fa = fa[index_flag].flatten()
    fb = fb[index_flag].flatten()
    fc = fc[index_flag].flatten()
    fd = fd[index_flag].flatten()

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    k0000 = ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
    p0000 = weight[k0000]#.reshape((-1, out_c,upscale,upscale))
    k0000 = None

    k0001 = ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
    p0001 = weight[k0001]#.reshape((-1, out_c,upscale,upscale))
    k0001 = None

    k0010 = ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
    p0010 = weight[k0010]#.reshape((-1, out_c,upscale,upscale))
    k0010 = None

    k0011 = ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
    p0011 = weight[k0011]#.reshape((-1, out_c,upscale,upscale))
    k0011 = None

    k0100 = ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
    p0100 = weight[k0100]#.reshape((-1, out_c,upscale,upscale))
    k0100 = None

    k0101 = ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
    p0101 = weight[k0101]#.reshape((-1, out_c,upscale,upscale))
    k0101 = None
    
    k0110 = ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
    p0110 = weight[k0110]#.reshape((-1, out_c,upscale,upscale))
    k0110 = None

    k0111 = ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
    p0111 = weight[k0111]#.reshape((-1, out_c,upscale,upscale))
    k0111 = None

    k1000 = ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
    p1000 = weight[k1000]#.reshape((-1, out_c,upscale,upscale))
    k1000 = None

    k1001 = ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
    p1001 = weight[k1001]#.reshape((-1, out_c,upscale,upscale))
    k1001 = None

    k1010 = ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
    p1010 = weight[k1010]#.reshape((-1, out_c,upscale,upscale))
    k1010 = None

    k1011 = ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
    p1011 = weight[k1011]#.reshape((-1, out_c,upscale,upscale))
    k1011 = None

    k1100 = ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
    p1100 = weight[k1100]#.reshape((-1, out_c,upscale,upscale))
    k1100 = None

    k1101 = ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
    p1101 = weight[k1101]#.reshape((-1, out_c,upscale,upscale))
    k1101 = None

    k1110 = ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
    p1110 = weight[k1110]#.reshape((-1, out_c,upscale,upscale))
    k1110 = None

    k1111 = ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
    p1111 = weight[k1111]#.reshape((-1, out_c,upscale,upscale))
    k1111 = None
    
    
    # Output image holder
    out = np.zeros((img_a1.shape[0],out_c*upscale*upscale))
    sz = img_a1.shape[0]
    # out = out.reshape(sz, -1)


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

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    # print(p0000[i].shape,fa[i].shape,i.shape,out_c)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]) / q
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]) / q
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]) / q
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q

    # print(out.dtype, out.max(),out.min(), q)
    # out = out / q
    return out,index_flag


# 4D equivalent of triangular interpolation, faster version
def FourSimplexInterpFaster_compress(weight_c1,weight_c2, img_in, h, w, interval, rot, compression_type, dw, delta_bit, upscale=4, out_c=1, mode='s',ref2index=None):
    if compression_type == 'xy':
        out_compress1, index_flag = InterpTorchBatch_compress_xy(weight_c1, img_in, h, w, interval, rot, dw, upscale,
                                                                   out_c,mode, ref2index=ref2index)
    elif compression_type == 'xyz':
        out_compress1, index_flag = InterpTorchBatch_compress_xyz(weight_c1, img_in, h, w, interval, rot, dw, upscale,
                                                                 out_c, mode, ref2index=ref2index)
    else:
        out_compress1, index_flag = InterpTorchBatch_compress_xyzt(weight_c1, img_in, h, w, interval, rot, dw, upscale,
                                                                   out_c,mode, ref2index=ref2index)
    index_flag = index_flag.flatten()

    interval = interval + delta_bit
    q = 2 ** interval 
    L = 2 ** (8 - interval) + 1 

    # weight_c2 = weight_c2.reshape(L*L*L*L,-1)

    img_abcd = (img_in // q).astype(np.int_)
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

    sz0, sz1, sz2, sz3 = img_a1.shape
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
    out_all = out.reshape(sz, -1)
    out = out_all[~index_flag]
    sz = out.shape[0]
    # print(sz)
    if sz == 0:
        out_all[index_flag] = out_compress1
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
        out_all = np.transpose(out_all, (0, 1, 4, 2, 5, 3, 6)).reshape(
            (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
        out_all = np.rot90(out_all, rot, [2, 3])
        return out_all
    # print(sz)

    img_a1 = img_a1.flatten()[~index_flag]
    img_b1 = img_b1.flatten()[~index_flag]
    img_c1 = img_c1.flatten()[~index_flag]
    img_d1 = img_d1.flatten()[~index_flag]

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    p0000 = weight_c2[
        img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0001 = weight_c2[
        img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0010 = weight_c2[
        img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0011 = weight_c2[
        img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0100 = weight_c2[
        img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0101 = weight_c2[
        img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0110 = weight_c2[
        img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p0111 = weight_c2[
        img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))

    p1000 = weight_c2[
        img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1001 = weight_c2[
        img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1010 = weight_c2[
        img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1011 = weight_c2[
        img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1100 = weight_c2[
        img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1101 = weight_c2[
        img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1110 = weight_c2[
        img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))
    p1111 = weight_c2[
        img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()]#.reshape(
        #(sz, out_c * upscale * upscale))

    # Output image holder
    # out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3],out_c, upscale, upscale))
    # sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]*img_a1.shape[3]
    # # out = out.reshape(sz, -1)
    # out_all = out.reshape(sz, -1)
    # out = out_all[~index_flag]

    # p0000 = p0000.reshape(sz, -1)
    # p0100 = p0100.reshape(sz, -1)
    # p1000 = p1000.reshape(sz, -1)
    # p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)[~index_flag]

    # p0001 = p0001.reshape(sz, -1)
    # p0101 = p0101.reshape(sz, -1)
    # p1001 = p1001.reshape(sz, -1)
    # p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)[~index_flag]
    fc = fc.reshape(-1, 1)[~index_flag]

    # p0010 = p0010.reshape(sz, -1)
    # p0110 = p0110.reshape(sz, -1)
    # p1010 = p1010.reshape(sz, -1)
    # p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)[~index_flag]

    # p0011 = p0011.reshape(sz, -1)
    # p0111 = p0111.reshape(sz, -1)
    # p1011 = p1011.reshape(sz, -1)
    # p1111 = p1111.reshape(sz, -1)

    fab = fa > fb;
    fac = fa > fc;
    fad = fa > fd

    fbc = fb > fc;
    fbd = fb > fd;
    fcd = fc > fd

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    # out = out / q
    # print(out.dtype, out.max(),out.min())
    out_all[index_flag] = out_compress1
    out_all[~index_flag] = out
    out = out_all
    out = out.reshape((sz0,sz1,sz2,sz3, out_c, upscale, upscale))
    out = np.transpose(out, (0,1,4,2,5,3,6)).reshape(
        (sz0, sz1*out_c, sz2 * upscale, sz3 * upscale))
    out = np.rot90(out, rot, [2, 3])

    return out

def InterpTorchBatch_channel(weight,out_c, img_in, interval):

    interval = interval
    q = 2 ** interval  # 16
    L = 2 ** (8 - interval) + 1  # 17

    # pytorch 1.5 dont support rounding_mode, use // equavilent
    # https://pytorch.org/docs/1.5.0/torch.html#torch.div
    tmp = (img_in // q).astype(np.uint8)
    # print(tmp.max(),tmp.min())
    img_a1,img_b1,img_c1,img_d1 = tmp[:,0:1,:,:],tmp[:,1:2,:,:],tmp[:,2:3,:,:],tmp[:,3:,:,:]

    # Extract LSBs
    tmp = (img_in % q).astype(np.int16)
    fa,fb,fc,fd = tmp[:,0:1,:,:],tmp[:,1:2,:,:],tmp[:,2:3,:,:],tmp[:,3:,:,:]


    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1
    weight = weight.reshape((L,L,L,L,-1))

    p0000 = weight[
        img_a1.flatten(), img_b1.flatten(), img_c1.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0001 = weight[
        img_a1.flatten(), img_b1.flatten(), img_c1.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0010 = weight[
        img_a1.flatten(), img_b1.flatten(), img_c2.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0011 = weight[
        img_a1.flatten(), img_b1.flatten(), img_c2.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0100 = weight[
        img_a1.flatten(), img_b2.flatten(), img_c1.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0101 = weight[
        img_a1.flatten(), img_b2.flatten(), img_c1.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0110 = weight[
        img_a1.flatten(), img_b2.flatten(), img_c2.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p0111 = weight[
        img_a1.flatten(), img_b2.flatten(), img_c2.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

    p1000 = weight[
        img_a2.flatten(), img_b1.flatten(), img_c1.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1001 = weight[
        img_a2.flatten(), img_b1.flatten(), img_c1.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1010 = weight[
        img_a2.flatten(), img_b1.flatten(), img_c2.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1011 = weight[
        img_a2.flatten(), img_b1.flatten(), img_c2.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1100 = weight[
        img_a2.flatten(), img_b2.flatten(), img_c1.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1101 = weight[
        img_a2.flatten(), img_b2.flatten(), img_c1.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1110 = weight[
        img_a2.flatten(), img_b2.flatten(), img_c2.flatten(), img_d1.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
    p1111 = weight[
        img_a2.flatten() , img_b2.flatten(), img_c2.flatten(), img_d2.flatten()].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
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

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = ((q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i])/q

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i])/q

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = ((q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i])/q
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = ((q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = ((q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i])/q

    # out = out / q
    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                               img_a1.shape[3], out_c))
    out = np.transpose(out,(0,1,4,2,3)).reshape(
        (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))
    return out

class eltr:
    def __init__(self, dataset, opt, lutDict,ref2index = None):
        folder = os.path.join(opt.testDir, dataset, 'HR')
        files = os.listdir(folder)
        files.sort()

        exp_name = opt.expDir.split("/")[-1]
        result_path = os.path.join(opt.resultRoot, exp_name, dataset, "X{}".format(opt.scale))
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        self.result_path = result_path
        self.dataset = dataset
        self.files = files
        self.opt = opt
        self.lutDict = lutDict
        self.ref2index = ref2index

    def run(self, num_worker=24):
        pool = Pool(num_worker)
        psnr_ssim_s = pool.map(self._worker, list(range(len(self.files))))
        print('Dataset {} | AVG LUT PSNR: {:.2f} SSIM: {:.4f}'.format(dataset, np.mean(np.asarray(psnr_ssim_s)[:, 0]),
                                                                      np.mean(np.asarray(psnr_ssim_s)[:, 1])))

    def _worker(self, i):
        # Load LR image
        img_lr = np.array(Image.open(
            os.path.join(self.opt.testDir, self.dataset, 'LR_bicubic/X{}'.format(self.opt.scale), self.files[i]))).astype(
            np.float32)
        if len(img_lr.shape) == 2:
            img_lr = np.expand_dims(img_lr, axis=2)
            img_lr = np.concatenate([img_lr, img_lr, img_lr], axis=2)
        # Load GT image
        img_gt = np.array(Image.open(os.path.join(self.opt.testDir, self.dataset, 'HR', self.files[i])))
        img_gt = modcrop(img_gt, self.opt.scale)
        out_c_list = [2, 2, 2, 1]
        refine_list = []

        if len(img_gt.shape) == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
            img_gt = np.concatenate([img_gt, img_gt, img_gt], axis=2)

        # conv1~4
        for s in range(4):
            stage = s + 1
            pred = 0
            for mode in self.opt.modes:
                if mode in ["d", "y"]:
                    pad = (0, 2)
                else:
                    pad = (0, 1)
                key1 = "s{}c0_{}_compress1".format(str(s + 1), mode)
                key2 = "s{}c0_{}_compress2".format(str(s + 1), mode)
                # key = "s{}c0_{}".format(str(s + 1), mode)

                for r in [0, 1, 2, 3]:
                    img_lr_rot = np.rot90(img_lr, r)
                    h, w, _ = img_lr_rot.shape
                    img_in = np.pad(img_lr_rot, (pad, pad, (0, 0)), mode='edge').transpose((2, 0, 1)) # (3, H, W)
                    img_in = img_in[:,np.newaxis,:,:] # (3, 1, H, W)
                    pred += FourSimplexInterpFaster_compress(self.lutDict[key1], self.lutDict[key2], img_in, h, w,
                                                    self.opt.interval, 4 - r, self.opt.cd, self.opt.dw, self.opt.si-self.opt.interval, upscale=1,out_c=out_c_list[s], mode=mode,
                                                    ref2index=self.ref2index) # (3,out_c,H,W)

            avg_factor, bias = len(self.opt.modes) * 4, 127
            img_lr = np.clip((pred / avg_factor) + bias, 0, 255)

            if out_c_list[s] == 2:
                x1, x2 = img_lr[:,0,:,:], img_lr[:,1,:,:] # (3, H, W)

                x2 = x2.transpose((1, 2, 0)) # (H, W, 3)
                x2 = np.round(np.clip(x2, 0, 255)).astype(np.float32)
                img_lr = x2

                x1 = x1[:,np.newaxis,:,:]
                x1 = np.round(np.clip(x1, 0, 255)).astype(np.float32)
                refine_list.append(x1)
            else:
                img_lr = np.round(np.clip(img_lr, 0, 255)).astype(np.float32)
                refine_list.append(img_lr)

        x = np.concatenate(refine_list,axis=1) # (3, 4, H, W)
        # conv5
        key = "s{}_channel".format(5)
        weight = self.lutDict[key]
        x = InterpTorchBatch_channel(weight, 4, x,self.opt.interval)
        x = np.round(np.clip(x+127, 0, 255)).astype(np.float32)

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:, c, :, :]
            x_c = x_c.transpose((1, 2, 0))
            for mode in self.opt.modes:
                if mode in ["d", "y"]:
                    pad = (0, 2)
                else:
                    pad = (0, 1)
                key1 = "s6c{}_{}_compress1".format(c, mode)
                key2 = "s6c{}_{}_compress2".format(c, mode)
                # key = "s6c{}_{}".format(c, mode)
                scale = self.opt.scale
                for r in [0, 1, 2, 3]:
                    img_lr_rot = np.rot90(x_c, r)
                    h, w, _ = img_lr_rot.shape
                    img_in = np.pad(img_lr_rot, (pad, pad, (0, 0)), mode='edge').transpose((2, 0, 1))
                    img_in = img_in[:, np.newaxis, :, :]
                    pred += FourSimplexInterpFaster_compress(self.lutDict[key1], self.lutDict[key2], img_in, h, w,
                                                    self.opt.interval, 4 - r, self.opt.cd, self.opt.dw, self.opt.si-self.opt.interval, upscale=scale, out_c=1,
                                                    mode=mode,ref2index=self.ref2index)  # (3,out_c,H,W)

        pred = pred[:,0,:,:] / 4
        avg_factor, bias = len(self.opt.modes), 0
        img_lr = np.clip((pred / avg_factor) + bias, 0, 255)
        img_lr = img_lr.transpose((1, 2, 0))
        img_lr = np.round(np.clip(img_lr, 0, 255)).astype(np.uint8)

        # Save to file
        img_out = img_lr
        Image.fromarray(img_out).save(
            os.path.join(self.result_path, '{}_{}_{}bit.png'.format(self.files[i].split('/')[-1][:-4], self.opt.lutName,
                                                                    8 - self.opt.interval)))
        y_gt, y_out = _rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0]
        psnr = PSNR(y_gt, y_out, self.opt.scale)
        ssim = cal_ssim(y_gt, y_out)
        return [psnr, ssim]


if __name__ == "__main__":
    opt = TestOptions().parse()
    L = 2 ** (8 - opt.interval) + 1  # 17
    # Load LUT
    lutDict = dict()

    compression_type = opt.cd

    for mode in opt.modes:
        # conv1

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        key = "s{}c0_{}_compress1".format(1, mode)
        if compression_type =='xy':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1,L,L, 2))
        elif compression_type == 'xyz':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, 2))
        else:
            lutDict[key] = np.load(lut_path).reshape(-1, 2)

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        key = "s{}c0_{}_compress2".format(1, mode)
        lutDict[key] = np.load(lut_path).reshape(-1, 2)

        # conv2

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        key = "s{}c0_{}_compress1".format(2, mode)
        if compression_type == 'xy':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, L, 2))
        elif compression_type == 'xyz':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, 2))
        else:
            lutDict[key] = np.load(lut_path).reshape(-1, 2)

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        key = "s{}c0_{}_compress2".format(2, mode)
        lutDict[key] = np.load(lut_path).reshape(-1, 2)

        # conv3


        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 3, mode))
        key = "s{}c0_{}_compress1".format(3, mode)
        if compression_type == 'xy':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, L, 2))
        elif compression_type == 'xyz':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, 2))
        else:
            lutDict[key] = np.load(lut_path).reshape(-1, 2)

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 3, mode))
        key = "s{}c0_{}_compress2".format(3, mode)
        lutDict[key] = np.load(lut_path).reshape(-1, 2)

        # conv4

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        key = "s{}c0_{}_compress1".format(4, mode)
        if compression_type == 'xy':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, L, 1))
        elif compression_type == 'xyz':
            lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, 1))
        else:
            lutDict[key] = np.load(lut_path).reshape(-1, 1)

        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        key = "s{}c0_{}_compress2".format(4, mode)
        lutDict[key] = np.load(lut_path).reshape(-1, 1)

        for c in range(4):
            # conv6

            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6,c, mode))
            key = "s{}c{}_{}_compress1".format(6,c, mode)
            if compression_type == 'xy':
                lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, L, opt.scale**2))
            elif compression_type == 'xyz':
                lutDict[key] = np.load(lut_path).astype(np.float32).reshape((-1, L, opt.scale**2))
            else:
                lutDict[key] = np.load(lut_path).reshape(-1, opt.scale**2)

            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6, c, mode))
            key = "s{}c{}_{}_compress2".format(6, c, mode)
            lutDict[key] = np.load(lut_path).reshape(-1, opt.scale ** 2)
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    key = "s{}_channel".format(5)
    lutDict[key] = np.load(lut_path).reshape(-1, 4)

    all_datasets = ['Set5', 'Set14','B100']
    # all_datasets = ['Urban100','Manga109']


    ref2index = np.load(os.path.join(opt.expDir,'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)))
    # ref2index = None

    for dataset in all_datasets:
        etr = eltr(dataset, opt, lutDict,ref2index)
        etr.run(8)



