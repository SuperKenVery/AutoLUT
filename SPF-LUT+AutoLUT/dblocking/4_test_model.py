import numpy as np
import sys

sys.path.insert(0, "../")  # run under the project directory
from common.option import TestOptions
from common.utils import PSNR, cal_ssim, bPSNR
from PIL import Image
import os
import torch
import model as Model
from data import DBBenchmark



def valid_steps(opt, model_G, valid, exp_name):
    datasets = ['classic5', 'LIVE1']
    _im_num = [5, 29]

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            ssims = []
            bpsnrs = []
            files = valid.files[datasets[i]]

            result_path = os.path.join(opt.resultRoot, exp_name, datasets[i], "qf{}".format(opt.qf))
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
                ssims.append(cal_ssim(pred[:, :, 0], lb[:, :, 0]))
                bpsnrs.append(bPSNR(pred, lb, 0))

                input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                Image.fromarray(input_img[:, :, 0]).save(
                    os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                Image.fromarray(lb[:, :, 0].astype(np.uint8)).save(
                    os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))

                Image.fromarray(pred[:, :, 0]).save(
                    os.path.join(result_path, '{}_net.png'.format(key.split('_')[-1])))
            print('Dataset {} | AVG Val PSNR: {:02f} SSIM: {:02f} bPSNR: {:02f}'.format(datasets[i],
                                                                                        np.mean(np.asarray(psnrs)),
                                                                                        np.mean(np.asarray(ssims)),
                                                                                        np.mean(np.asarray(bpsnrs))))


if __name__ == '__main__':
    opt_inst = TestOptions()
    opt = opt_inst.parse()


    if opt.model in ['BaseDNNets', 'SPF_LUT_net']:
        model = getattr(Model, opt.model)
        model_G = model(nf=opt.nf, modes=opt.modes, stages=opt.stages).cuda()
        lm = torch.load(opt.load_model_path)
        model_G.load_state_dict(lm, strict=True)
    elif opt.model in ['BaseMuLUT_DFC', 'SPF_LUT_DFC']:
        model = getattr(Model, opt.model)
        model_G = model(lut_folder=opt.expDir, modes=opt.modes, stages=opt.stages, lutName=opt.lutName, qf=opt.qf, interval=opt.interval,
                    compressed_dimensions=opt.cd, diagonal_width=opt.dw, sampling_interval=opt.si,
                    c_in=1).cuda()

    # Valid dataset
    valid = DBBenchmark(opt.testDir, qf=opt.qf)

    valid_steps(opt, model_G, valid, opt.expDir.split('/')[-1])
