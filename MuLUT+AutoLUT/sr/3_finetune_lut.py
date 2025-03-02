from loguru import logger
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import model
from data import Provider, SRBenchmark, DebugDataProvider, DebugDataset

sys.path.insert(0, "../")  # run under the current directory
from common.option import TrainOptions
from common.utils import PSNR, cal_ssim, logger_info, _rgb2ycbcr
from common.network import LrLambda

torch.backends.cudnn.benchmark = True


def valid_steps(model_G, valid, opt, iter):
    if opt.debug:
        datasets = ['Set5', 'Set14']
    else:
        datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs, ssims = [], []
            files = valid.files[datasets[i]]

            result_path = os.path.join(opt.valoutDir, datasets[i])
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = torch.tensor(valid.ims[key])
                input_im = valid.ims[key + 'x%d' % opt.scale]

                input_im = input_im.astype(np.float32) / 255.0
                im = torch.tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()

                pred = model_G(im) * 255.0

                pred = np.transpose(np.squeeze(
                    pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)
                pred = torch.tensor(pred)

                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(np.array(PSNR(left, right, opt.scale)))
                ssims.append(np.array(cal_ssim(left, right)))

                Image.fromarray(pred.cpu().detach().numpy()).save(
                    os.path.join(result_path, '{}_lutft.png'.format(key.split('_')[-1])))

            logger.info('Iter {} | Dataset {} | AVG PSNR: {:02f}, AVG: SSIM: {:04f}'.format(iter, datasets[i],
                                                                                            np.mean(np.asarray(psnrs)),
                                                                                            np.mean(np.asarray(ssims))))
            logger.info(f"Per-image PSNR: {np.asarray(psnrs)}")

def save(model_G, opt, lr_sched):
    # Save finetuned LUTs
    for s in range(stages):
        stage = s + 1
        for sampler in range(numSamplers):
            # MuLUTUnit, LUT
            ft_lut_path = os.path.join(opt.expDir,
                                       "LUT_ft_x{}_{}bit_int8_s{}_{}.npy".format(opt.scale, opt.interval, s,sampler))
            lut_weight = np.round(np.clip(getattr(model_G, "weight_s{}_{}".format(str(s), sampler)).cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
            np.save(ft_lut_path, lut_weight)

            # AutoSampler
            key = "s{}_{}".format(str(s), sampler)
            sampler_path = os.path.join(opt.expDir, f"LUT_ft_x{opt.scale}_{opt.interval}bit_int8_s{s}_{sampler}_sampler.npy")
            sampler_weights=getattr(model_G, key+"_sampler").sampler.weight.cpu().detach().numpy()
            np.save(sampler_path, sampler_weights)

            # Residual
            if s>0:
                residual_key = key+"_residual"
                residual_path = os.path.join(opt.expDir, f"LUT_ft_x{opt.scale}_{opt.interval}bit_int8_s{s}_{sampler}_residual.npy")
                residual_weights=getattr(model_G, residual_key).cpu().detach().numpy()
                np.save(residual_path, residual_weights)

    if lr_sched != None:
        torch.save(lr_sched, os.path.join(
            opt.expDir, 'LRSched_{:06d}.pth'.format(i)))


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    logger_name = 'lutft'
    logger_info(opt.debug,os.path.join(opt.expDir, 'lutft-{time}.log'))
    logger.info(opt_inst.describe_options(opt))

    numSamplers = opt.numSamplers
    stages = opt.stages

    model = model.AutoLUT

    model_G = model(lut_folder=opt.expDir, stages=stages, samplers=opt.numSamplers, sample_size=opt.sampleSize, upscale=opt.scale, interval=opt.interval).cuda()

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # Learning rate schedule
    initial_lr = 1 if opt.lambda_lr else opt.lr0
    opt_G = optim.Adam(params_G, lr=initial_lr, weight_decay=opt.weightDecay, amsgrad=False, fused=True)

    if opt.lambda_lr:
        lr_sched_obj=LrLambda(opt)
        lr_sched = optim.lr_scheduler.LambdaLR(opt_G, lr_sched_obj)

    # Load saved params
    if opt.startIter > 0:
        lm = torch.load(
            opt.expDir, 'Model_{:06d}.pth'.format(opt.startIter))
        model_G.load_state_dict(lm.state_dict(), strict=True)

        lm = torch.load(opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter))
        opt_G.load_state_dict(lm.state_dict())

        if opt.lambda_lr:
            lm = torch.load(
                os.path.join(opt.expDir, 'LRSched_{:06d}.pth'.format(opt.startIter))
            )
            lr_sched.load_state_dict(lm.state_dict())

    # Training dataset
    if not opt.debug:
        train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize)
    else:
        train_iter = DebugDataProvider(DebugDataset())

    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    # Training
    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0
    i = opt.startIter
    for i in tqdm(range(opt.startIter + 1, opt.totalIter + 1), dynamic_ncols=True):
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb = train_iter.next()
        im = im.cuda()
        lb = lb.cuda()
        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()

        pred = model_G(im)

        loss_G = F.mse_loss(pred, lb)
        loss_G.backward()
        opt_G.step()
        if opt.lambda_lr:
            lr_sched.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        # Show information
        if i % opt.displayStep == 0:
            logger.info("{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                opt.expDir, i, accum_samples, l_accum[0] / opt.displayStep, dT / opt.displayStep,
                                              rT / opt.displayStep))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        # Save
        if i % opt.saveStep == 0:
            save(model_G, opt, lr_sched if opt.lambda_lr else None)

        # Validation
        if opt.debug==False and ((i % opt.valStep == 0) or (i == 1)):
            # Validation during multi GPU training
            if opt.gpuNum > 1:
                valid_steps(model_G.module, valid, opt, i)
            else:
                valid_steps(model_G, valid, opt, i)


    save(model_G, opt, lr_sched if opt.lambda_lr else None)

    logger.info("Finetuned LUT saved to {}".format(opt.expDir))
    logger.info("Complete")
