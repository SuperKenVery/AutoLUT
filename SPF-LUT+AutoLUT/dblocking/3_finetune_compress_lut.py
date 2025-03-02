import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

import model as Model
from data import Provider, DBBenchmark

sys.path.insert(0, "../")  # run under the project directory
from common.option import TrainOptions
from common.utils import logger_info, PSNR
from common.Writer import Logger

torch.backends.cudnn.benchmark = True

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


def mulut_predict(model_G, x, phase="train", opt=None):
    modes, stages = opt.modes, opt.stages
    # Stage 1
    for s in range(stages):
        pred = 0
        for mode in modes:
            pad = mode_pad_dict[mode]
            for r in [0, 1, 2, 3]:
                pred += round_func(torch.rot90(model_G(F.pad(torch.rot90(x, r, [
                    2, 3]), (0, pad, 0, pad), mode='replicate'), stage=s + 1, mode=mode), (4 - r) % 4, [2, 3]) * 127)
        if s + 1 == stages:
            avg_factor, bias, norm = len(modes), 0, 1
            x = round_func((pred / avg_factor) + bias)
            if phase == "train":
                x = x / 255.0
        else:
            avg_factor, bias, norm = len(modes) * 4, 127, 255.0
            x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm

    return x


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def SaveCheckpoint(model_G, opt_G, opt, i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, os.path.join(
        opt.expDir, 'Model_{:06d}{}.pth'.format(i, str_best)))
    torch.save(opt_G, os.path.join(
        opt.expDir, 'Opt_{:06d}{}.pth'.format(i, str_best)))
    logger.info("Checkpoint saved {}".format(str(i)))


def valid_steps(model_G, valid, opt, iter):
    datasets = ['classic5', 'LIVE1']
    _im_num = [5, 29]
    psnr_list = []

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            files = valid.files[datasets[i]]

            result_path = os.path.join(opt.valoutDir, datasets[i])
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = valid.ims[key]
                input_im = valid.ims[key + 'x%d' % opt.qf]

                # no need to divide 255.0
                # input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()

                pred = model_G(im, 'valid')

                pred = np.transpose(np.squeeze(
                    pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)
                psnrs.append(
                    PSNR(pred[:, :, 0], lb[:, :, 0], 0))  # single channel, no scale change

                if iter < 10000:  # save input and gt at start
                    input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                    Image.fromarray(input_img[:,:,0]).save(
                        os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                    Image.fromarray(lb[:,:,0].astype(np.uint8)).save(
                        os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))

                Image.fromarray(pred[:,:,0]).save(
                    os.path.join(result_path, '{}_net.png'.format(key.split('_')[-1])))

            logger.info(
                'Iter {} | Dataset {} | AVG Val PSNR: {:02f}'.format(iter, datasets[i], np.mean(np.asarray(psnrs))))
            writer.scalar_summary('PSNR_valid/{}'.format(datasets[i]), np.mean(np.asarray(psnrs)), iter)
            psnr_list.append(np.mean(np.asarray(psnrs)))
        return psnr_list


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    # Tensorboard for monitoring
    writer = Logger(log_dir=opt.logDir)

    logger_name = 'train'
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, opt.model)
    c_in = 1
    model_G = model(lut_folder=opt.expDir, modes=modes, stages=stages, qf=opt.qf, interval=opt.interval,
                    compressed_dimensions=opt.cd, diagonal_width=opt.dw, sampling_interval=opt.si,
                    c_in=1).cuda()

    if opt.gpuNum > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=list(range(opt.gpuNum)))

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # LR
    if opt.lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Load saved params
    if opt.startIter > 0:
        lm = torch.load(
            os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.startIter)))
        model_G.load_state_dict(lm, strict=True)

        lm = torch.load(os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter)))
        opt_G.load_state_dict(lm.state_dict())

    # Training dataset
    train_iter = Provider(opt.batchSize, opt.workerNum, opt.qf, opt.trainDir, opt.cropSize)

    # Valid dataset
    valid = DBBenchmark(opt.valDir, qf=opt.qf)

    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter
    for i in range(opt.startIter + 1, opt.totalIter + 1):
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb = train_iter.next()
        im = im.cuda()
        lb = lb.cuda()
        dT += time.time() - st

        # TRAIN G
        st = time.time()
        opt_G.zero_grad()

        pred = model_G(im, 'train')

        loss_G = F.mse_loss(pred, lb)
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        # Show information
        if i % opt.displayStep == 0:
            writer.scalar_summary('loss_Pixel', l_accum[0] / opt.displayStep, i)

            logger.info("{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                opt.expDir, i, accum_samples, l_accum[0] / opt.displayStep, dT / opt.displayStep,
                                              rT / opt.displayStep))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        # Validation
        if i % opt.valStep == 0:
            # validation during multi GPU training
            if opt.gpuNum > 1:
                psnr_list = valid_steps(model_G.module, valid, opt, i)
            else:
                psnr_list = valid_steps(model_G, valid, opt, i)


    # Save finetuned LUTs
    if opt.model in ['SPF_LUT_DFC', 'SPF_LUT']:
        for k,v in model_G.named_parameters():
            ft_lut_path = os.path.join(opt.expDir,"{}.npy".format(k))
            lut_weight = np.round(np.clip(v.cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
            np.save(ft_lut_path, lut_weight)
    else:
        for s in range(stages):
            stage = s + 1
            for c in range(c_in):
                for mode in modes:
                    ft_lut_path = os.path.join(opt.expDir,"LUT_ft_qf{}_{}bit_int8_s{}_c{}_{}_compress1.npy".format(opt.qf, opt.interval,str(stage),str(c), mode))
                    lut_weight = np.round(np.clip(getattr(model_G, "weight_s{}_c{}_{}_compress1".format(str(s + 1), str(c), mode)).cpu().detach().numpy(),-1, 1) * 127).astype(np.int8)
                    np.save(ft_lut_path, lut_weight)

                    ft_lut_path = os.path.join(opt.expDir,"LUT_ft_qf{}_{}bit_int8_s{}_c{}_{}_compress2.npy".format(opt.qf,opt.interval,str(stage),str(c), mode))
                    lut_weight = np.round(np.clip(getattr(model_G,"weight_s{}_c{}_{}_compress2".format(str(s + 1), str(c), mode)).cpu().detach().numpy(),-1, 1) * 127).astype(np.int8)
                    np.save(ft_lut_path, lut_weight)

    logger.info("Finetuned LUT saved to {}".format(opt.expDir))
    logger.info("Complete")
