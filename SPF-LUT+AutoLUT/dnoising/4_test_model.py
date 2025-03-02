import numpy as np
import sys
sys.path.insert(0, "../")  # run under the project directory
from common.option import TestOptions
from common.utils import cPSNR,PSNR
from PIL import Image
import os
import torch
import model as Model
from data import DNBenchmark

def valid_steps(opt, model_G, valid,exp_name):
    datasets = ['Set12', 'BSD68']
    # datasets = ['BSD68']

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            files = valid.files[datasets[i]]

            # result_path = os.path.join(valoutDir, datasets[i])
            result_path = os.path.join(opt.resultRoot, exp_name, datasets[i], "sigma{}".format(opt.sigma))
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = valid.ims[key]
                input_im = valid.ims[key + 'x%d' % opt.sigma]

                # no need to divide 255.0
                # input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                # im = torch.clamp(im,0,1)
                pred = model_G(im, 'valid')

                pred = np.transpose(np.squeeze(
                    pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                left, right = pred[:, :, 0], lb[:, :, 0]
                psnrs.append(PSNR(left, right, 0))  # single channel, no scale change

                input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                Image.fromarray(input_img[:,:,0]).save(
                    os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                Image.fromarray(lb[:,:,0].astype(np.uint8)).save(
                    os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))

                Image.fromarray(pred[:,:,0]).save(
                    os.path.join(result_path, '{}_net.png'.format(key.split('_')[-1])))

            print('Dataset {} | AVG Val PSNR: {:02f}'.format(datasets[i], np.mean(np.asarray(psnrs))))

# def transfer_pth(opt):
#     model = getattr(model_new, opt.model)
#     model_G = model(nf=opt.nf, modes=opt.modes, stages=opt.stages).cuda()
#     lm = torch.load(opt.load_model_path)
#     model_G.load_state_dict(lm.state_dict(), strict=True)
#     torch.save(model_G.state_dict(), os.path.join(opt.expDir, 'best_model.pth'))

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
        model_G = model(lut_folder=opt.expDir, modes=opt.modes, stages=opt.stages, lutName=opt.lutName, sigma=opt.sigma, interval=opt.interval,
                    compressed_dimensions=opt.cd, diagonal_width=opt.dw, sampling_interval=opt.si,
                    c_in=1).cuda()

    # Valid dataset
    valid = DNBenchmark(opt.testDir, sigma=opt.sigma)

    valid_steps(opt, model_G, valid, opt.expDir.split('/')[-1])
