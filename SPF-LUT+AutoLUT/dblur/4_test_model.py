import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, "../")  # run under the project directory
from common.option import TestOptions
from common.utils import PSNR, cal_ssim,_rgb2ycbcr
from PIL import Image
import os
import torch
import model as Model
from data import GoProTest


def valid_steps(opt, model_G, valid, exp_name):
    with torch.no_grad():
        model_G.eval()

        psnrs = []
        ssims = []

        result_path = os.path.join(opt.resultRoot,exp_name)
        if not os.path.isdir(result_path):
            os.makedirs(result_path)

        for key in tqdm(range(len(valid))):
            input_im = valid.lr_ims['/gdata/liyl'+valid.blur_list[key][7:].replace('\\','/')]
            lb = valid.hr_ims['/gdata/liyl'+valid.sharp_list[key][7:].replace('\\','/')]
            im = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), 0).astype(np.float32) / 255.0).cuda()
            print(im.shape)

            pred = model_G(im, 'valid')

            pred = np.transpose(np.squeeze(
                pred.data.cpu().numpy(), 0), [1, 2, 0])
            pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

            # Y PSNR
            # left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
            # Color PSNR
            left, right = pred, lb
            psnrs.append(PSNR(left, right, 0))

            left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
            ssims.append(cal_ssim(left,right))

            input_img = np.round(np.clip(input_im, 0, 255)).astype(np.uint8)
            # Image.fromarray(input_img).save(
            #     os.path.join(result_path, valid.blur_list[key].replace("/", "_").replace(".png", "_input.png")))
            # Image.fromarray(lb.astype(np.uint8)).save(
            #     os.path.join(result_path, valid.sharp_list[key].replace("/", "_").replace(".png", "_gt.png")))
            Image.fromarray(pred).save(os.path.join(result_path, valid.sharp_list[key].replace("/", "_").replace(".png", "_net.png")))

        print('Dataset {} | AVG Val PSNR: {} SSIM: {}'.format("GoPro",np.mean(np.asarray(psnrs)),np.mean(np.asarray(ssims))))


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
        model_G = model(lut_folder=opt.expDir, modes=opt.modes, stages=opt.stages, lutName=opt.lutName, interval=opt.interval,
                    compressed_dimensions=opt.cd, diagonal_width=opt.dw, sampling_interval=opt.si,
                    c_in=1).cuda()

    # Valid dataset
    valid = GoProTest(opt.testDir)


    valid_steps(opt, model_G, valid, opt.expDir.split('/')[-1])
