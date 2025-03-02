

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob


from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter


### USER PARAMS ###
EXP_NAME = "SR-LUT-autosample"
VERSION = "S"
UPSCALE = 4     # upscaling factor
SAMPLE_SIZE = 3

NB_BATCH = 32        # mini-batch
CROP_SIZE = 48       # input LR training patch size

START_ITER = 116000      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER = 400000    # Total number of training iterations

I_DISPLAY = 100     # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './val/'      # Validation images

LR_G = 1e-4         # Learning rate for the generator

### Tensorboard for monitoring ###
writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))

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



### A lightweight deep network ###
class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale
        self.sampler = AutoSample(SAMPLE_SIZE)

        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    @staticmethod
    def unfold(x):
        B, C, H, W = x.shape
        x = F.unfold(x, SAMPLE_SIZE)
        P = SAMPLE_SIZE-1
        x = x.view(B, C, SAMPLE_SIZE*SAMPLE_SIZE, (H-P)*(W-P))
        x = x.permute((0,1,3,2))  # B, C, L, K*K
        x = x.reshape(B*C*(H-P)*(W-P), SAMPLE_SIZE, SAMPLE_SIZE)  # BCL, K, K
        x = x.unsqueeze(1)  # BCL, 1, K, K

        return x, (B, C, H, W)
    @staticmethod
    def put_back(x, ori_shape):
        B, C, H, W = ori_shape
        x = x.squeeze(1)
        P = SAMPLE_SIZE-1
        S = UPSCALE
        x = x.reshape(B, C, (H-P)*(W-P), -1)      # B, C, L, K*K
        x = x.permute((0,1,3,2))    # B, C, K*K, L
        x = x.reshape(B, -1, (H-P)*(W-P))   # B, CKK, L
        x = F.fold(
            x,
            output_size = ((H-P)*S, (W-P)*S),
            kernel_size = S,
            stride = S,
        )       # B, C, (H-P)*S, (W-P)*S
        assert x.shape==(B, C, (H-P)*S, (W-P)*S)

        return x

    def forward(self, x_in):
        B, C, H, W = x_in.size()

        unf, shape = self.unfold(x_in)
        sampled = self.sampler(unf)     # BCL, 1, 2, 2

        x = self.conv1(sampled)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)   # BCL, 1 2*upscale, 2*upscale
        x = self.put_back(x, shape)  # B, C, (H-P)*S, (W-P)*S

        return x

model_G = SRNet(upscale=UPSCALE).cuda()



## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)



## Load saved params
if START_ITER > 0:
    lm = torch.load('checkpoint/{}/model_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    opt_G.load_state_dict(lm.state_dict())


# Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K(
                                datadir = TRAIN_DIR,
                                crop_size = CROP_SIZE,
                                crop_per_image = NB_BATCH//4,
                                out_batch_size = NB_BATCH,
                                scale_factor = UPSCALE,
                                shuffle=True))
Iter_H.start(max_q_size=16, workers=4)


## Prepare directories
if not isdir('checkpoint'):
    mkdir('checkpoint')
if not isdir('result'):
    mkdir('result')
if not isdir('checkpoint/{}'.format(str(VERSION))):
    mkdir('checkpoint/{}'.format(str(VERSION)))
if not isdir('result/{}'.format(str(VERSION))):
    mkdir('result/{}'.format(str(VERSION)))




## Some preparations
print('===> Training start')
l_accum = [0.,0.,0.]
dT = 0.
rT = 0.
accum_samples = 0


def SaveCheckpoint(i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, 'checkpoint/{}/model_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
    torch.save(opt_G, 'checkpoint/{}/opt_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best))
    print("Checkpoint saved")



### TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1), dynamic_ncols=True):

    model_G.train()

    # Data preparing
    st = time.time()
    batch_L, batch_H = Iter_H.dequeue()
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()      # BxCxHxW, range [0,1]
    batch_L = Variable(torch.from_numpy(batch_L)).cuda()      # BxCxHxW, range [0,1]
    dT += time.time() - st


    ## TRAIN G
    st = time.time()
    opt_G.zero_grad()

    # Rotational ensemble training
    padding = (0, SAMPLE_SIZE-1, 0, SAMPLE_SIZE-1)
    batch_S1 = model_G(F.pad(batch_L, padding, mode='reflect'))

    batch_S2 = model_G(F.pad(torch.rot90(batch_L, 1, [2,3]), padding, mode='reflect'))
    batch_S2 = torch.rot90(batch_S2, 3, [2,3])

    batch_S3 = model_G(F.pad(torch.rot90(batch_L, 2, [2,3]), padding, mode='reflect'))
    batch_S3 = torch.rot90(batch_S3, 2, [2,3])

    batch_S4 = model_G(F.pad(torch.rot90(batch_L, 3, [2,3]), padding, mode='reflect'))
    batch_S4 = torch.rot90(batch_S4, 1, [2,3])


    batch_S = ( torch.clamp(batch_S1,-1,1)*127 + torch.clamp(batch_S2,-1,1)*127 )
    batch_S += ( torch.clamp(batch_S3,-1,1)*127 + torch.clamp(batch_S4,-1,1)*127 )
    batch_S /= 255.0

    loss_Pixel = torch.mean( ((batch_S - batch_H)**2)  )
    loss_G = loss_Pixel

    # Update
    loss_G.backward()
    opt_G.step()
    rT += time.time() - st

    # For monitoring
    accum_samples += NB_BATCH
    l_accum[0] += loss_Pixel.item()


    ## Show information
    if i % I_DISPLAY == 0:
        writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)

        tqdm.write("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.]
        dT = 0.
        rT = 0.


    ## Save models
    if i % I_SAVE == 0:
        SaveCheckpoint(i)


    ## Validation
    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_G.eval()

            # Test for validation images
            files_gt = glob.glob(VAL_DIR + '/HR/*.png')
            files_gt.sort()
            files_lr = glob.glob(VAL_DIR + '/LR/*.png')
            files_lr.sort()

            psnrs = []
            lpips = []

            for ti, fn in enumerate(files_gt):
                # Load HR image
                tmp = _load_img_array(files_gt[ti])
                val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

                # Load LR image
                tmp = _load_img_array(files_lr[ti])
                val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
                val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
                val_L = val_L[np.newaxis, ...]            # BxCxHxW

                val_L = Variable(torch.from_numpy(val_L.copy()), volatile=True).cuda()

                # Run model
                batch_S1 = model_G(F.pad(val_L, padding, mode='reflect'))

                batch_S2 = model_G(F.pad(torch.rot90(val_L, 1, [2,3]), padding, mode='reflect'))
                batch_S2 = torch.rot90(batch_S2, 3, [2,3])

                batch_S3 = model_G(F.pad(torch.rot90(val_L, 2, [2,3]), padding, mode='reflect'))
                batch_S3 = torch.rot90(batch_S3, 2, [2,3])

                batch_S4 = model_G(F.pad(torch.rot90(val_L, 3, [2,3]), padding, mode='reflect'))
                batch_S4 = torch.rot90(batch_S4, 1, [2,3])

                batch_S = ( torch.clamp(batch_S1,-1,1)*127 + torch.clamp(batch_S2,-1,1)*127 )
                batch_S += ( torch.clamp(batch_S3,-1,1)*127 + torch.clamp(batch_S4,-1,1)*127 )
                batch_S /= 255.0


                # Output
                image_out = (batch_S).cpu().data.numpy()
                image_out = np.clip(image_out[0], 0. , 1.)      # CxHxW
                image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC

                # Save to file
                image_out = ((image_out)*255).astype(np.uint8)
                Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

                # PSNR on Y channel
                img_gt = (val_H*255).astype(np.uint8)
                CROP_S = 4
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

        tqdm.write('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))

        writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
        writer.flush()
