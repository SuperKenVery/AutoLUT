
from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(64-5)  # Set according to your machine


import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr

class AutoSample(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_shape=input_size
        self.sampler=nn.Conv2d(1,4,input_size)
        self.shuffel=nn.PixelShuffle(2)
        self.nw=input_size**2

    def forward_sample(self, x):
        assert len(x.shape)==4 and x.shape[-2:]==(self.input_shape,self.input_shape), f"Unexpected shape: {x.shape}"
        # x = self.sampler(x)
        # logger.debug(self.sampler.weight)
        w = F.softmax(self.sampler.weight.view(-1, self.nw), dim=1).view_as(self.sampler.weight)
        x = F.conv2d(x, w, bias=self.sampler.bias*0)
        x = self.shuffel(x)
        return x

    def forward_nosample(self, x):
        assert len(x.shape)==4 and x.shape[-2:]==(self.input_shape,self.input_shape), f"Unexpected shape: {x.shape}"
        b,c,h,w = x.shape
        out=x[:,:,0:2,0:2]
        return out

    def forward(self, x):
        return self.forward_sample(x)


# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 3        # N bit uniform sampling
LUT_PATH = "Model_S_x{}_{}bit_int8.npy.me".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
# SAM_PATH = "Model_S_x4_autosample.npy"
SAM_PATH = "Model_S_x4_autosample.npy"
TEST_DIR = './Set5/'      # Test images
SAMPLE_SIZE = 3



# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
sampler = AutoSample(SAMPLE_SIZE)
sw=np.load(SAM_PATH)
sampler.sampler.weight=torch.nn.Parameter(torch.tensor(sw))



# Test LR images
files_lr = glob.glob(TEST_DIR + '/LR_x{}/*.png'.format(UPSCALE))
files_lr.sort()

# Test GT images
files_gt = glob.glob(TEST_DIR + '/HR/*.png')
files_gt.sort()


psnrs = []

if not isdir('./output_S_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL)):
    mkdir('./output_S_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL))

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

def put_back(P, x, ori_shape):
    B, C, H, W=ori_shape
    x = x.squeeze(1)
    x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
    x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
    x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
    x = F.fold(x, ((H - P), (W - P)),
               1, stride=1)
    return x

for ti, fn in enumerate(tqdm(files_gt)):
    # Load LR image
    img_lr = np.array(Image.open(files_lr[ti])).astype(np.float32)
    h, w, c = img_lr.shape

    # Load GT image
    img_gt = np.array(Image.open(files_gt[ti]))


    # Sampling interval for input
    q = 2**SAMPLING_INTERVAL


    # 4D equivalent of triangular interpolation
    def FourSimplexInterp(weight, img_in, h, w, q, rot, upscale=4):
        L = 2**(8-SAMPLING_INTERVAL) + 1

        # AutoSample
        img_in = np.expand_dims(img_in, axis=0)
        img_in = torch.tensor(img_in)

        unfolded, shape = unfold(sampler.input_shape, sampler.input_shape-1, img_in)
        assert unfolded.shape[-2:]==(sampler.input_shape, sampler.input_shape)

        sampled = sampler(unfolded)
        assert sampled.shape[:-2]==unfolded.shape[:-2] and sampled.shape[-2:]==(2,2)

        _a = sampled[:,:,0,0].unsqueeze(-1).unsqueeze(-1)
        _b = sampled[:,:,0,1].unsqueeze(-1).unsqueeze(-1)
        _c = sampled[:,:,1,0].unsqueeze(-1).unsqueeze(-1)
        _d = sampled[:,:,1,1].unsqueeze(-1).unsqueeze(-1)
        assert _a.shape==_b.shape==_c.shape==_d.shape

        a=put_back(sampler.input_shape-1, _a, shape)
        b=put_back(sampler.input_shape-1, _b, shape)
        c=put_back(sampler.input_shape-1, _c, shape)
        d=put_back(sampler.input_shape-1, _d, shape)

        a, b, c, d = a.detach().numpy(), b.detach().numpy(), c.detach().numpy(), d.detach().numpy(),
        a, b, c, d = a.squeeze(0), b.squeeze(0), c.squeeze(0), d.squeeze(0)

        # Extract MSBs
        img_a1 = a // q
        img_b1 = b // q
        img_c1 = c // q
        img_d1 = d // q

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        # Extract LSBs
        fa_ = a % q
        fb_ = b % q
        fc_ = c % q
        fd_ = d % q

        # Vertices (O in Eq3 and Table3 in the paper)
        p0000 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0001 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0010 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0011 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0100 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0101 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0110 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p0111 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

        p1000 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1001 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1010 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1011 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1100 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1101 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1110 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p1111 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

        # Output image holder
        out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

        # Naive pixelwise output value interpolation (Table3 in the paper)
        # It would be faster implemented with a parallel operation
        for c in range(img_a1.shape[0]):
            for y in range(img_a1.shape[1]):
                for x in range(img_a1.shape[2]):
                    fa = fa_[c,y,x]
                    fb = fb_[c,y,x]
                    fc = fc_[c,y,x]
                    fd = fd_[c,y,x]

                    if fa > fb:
                        if fb > fc:
                            if fc > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fb > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                            elif fa > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fa > fc:
                            if fb > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fc > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                            elif fa > fd:
                                out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        else:
                            if fb > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fc > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                            elif fa > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]

                    else:
                        if fa > fc:
                            if fc > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fa > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                            elif fb > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fb > fc:
                            if fa > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fc > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                            elif fb > fd:
                                out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        else:
                            if fa > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                            elif fb > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                            elif fc > fd:
                                out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                            else:
                                out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]

        out = np.transpose(out, (0, 1,3, 2,4)).reshape((img_a1.shape[0], img_a1.shape[1]*upscale, img_a1.shape[2]*upscale))
        out = np.rot90(out, rot, [1,2])
        out = out / q
        return out


    # Rotational ensemble
    padding = ((0, SAMPLE_SIZE-1), (0, SAMPLE_SIZE-1), (0,0))
    img_in = np.pad(img_lr, padding, mode='reflect').transpose((2,0,1))
    out_r0 = FourSimplexInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 1), padding, mode='reflect').transpose((2,0,1))
    out_r1 = FourSimplexInterp(LUT, img_in, w, h, q, 3, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 2), padding, mode='reflect').transpose((2,0,1))
    out_r2 = FourSimplexInterp(LUT, img_in, h, w, q, 2, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 3), padding, mode='reflect').transpose((2,0,1))
    out_r3 = FourSimplexInterp(LUT, img_in, w, h, q, 1, upscale=UPSCALE)

    img_out = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0) / 255.0
    img_out = img_out.transpose((1,2,0))
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    # Matching image sizes
    if img_gt.shape[0] < img_out.shape[0]:
        img_out = img_out[:img_gt.shape[0]]
    if img_gt.shape[1] < img_out.shape[1]:
        img_out = img_out[:, :img_gt.shape[1]]

    if img_gt.shape[0] > img_out.shape[0]:
        img_out = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
    if img_gt.shape[1] > img_out.shape[1]:
        img_out = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # Save to file
    Image.fromarray(img_out).save('./output_S_x{}_{}bit/{}_LUT_interp_{}bit.png'.format(UPSCALE, SAMPLING_INTERVAL, fn.split('/')[-1][:-4], SAMPLING_INTERVAL))

    CROP_S = 4
    psnr = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S)
    psnrs.append(psnr)

print('Test: {}, Sample Interval: {}, AVG PSNR: {}'.format(TEST_DIR, SAMPLING_INTERVAL, np.mean(np.asarray(psnrs))))

