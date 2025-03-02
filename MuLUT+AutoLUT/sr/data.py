import os
import random
import sys
import torch

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")  # run under the project directory
from common.utils import modcrop
from loguru import logger



class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True, debug=False,gpuNum=1):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 901)]  # use both train and valid
        self.debug=debug
        self.gpuNum=gpuNum

        if not debug and gpuNum==1:
            # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
            self.hr_cache = os.path.join(path, "cache_hr.npy")
            if not os.path.exists(self.hr_cache):
                self.cache_hr()
                logger.info("HR image cache to: {}", self.hr_cache)
            self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
            logger.info("HR image cache from: {}", self.hr_cache)

            self.lr_cache = os.path.join(path, "cache_lr_x{}.npy".format(self.scale))
            if not os.path.exists(self.lr_cache):
                self.cache_lr()
                logger.info("LR image cache to: {}", self.lr_cache)
            self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
            logger.info("LR image cache from: {}", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f + "x{}.png".format(self.scale))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f + ".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        if self.debug or self.gpuNum>1:
            dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
            im = np.array(Image.open(os.path.join(dataLR, key + "x{}.png".format(self.scale))))
            dataHR = os.path.join(self.path, "HR")
            lb = np.array(Image.open(os.path.join(dataHR, key + ".png")))
        else:
            lb = self.hr_ims[key]
            im = self.lr_ims[key]

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, c]
        im = im[i:i + self.sz, j:j + self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)  # High resolution
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)  # Low resolution

        return im, lb   # Input, Ground truth

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path_ori, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        self.path=path_ori
        self.scale=scale
        self.cache_files_path=os.path.join(self.path,'test_cache_files.npy')
        self.cache_imgs_path=os.path.join(self.path,'test_cache_imgs.npy')

        if os.path.exists(self.cache_files_path) and os.path.exists(self.cache_imgs_path):
            logger.info(f"Test data from cache {self.cache_files_path}, {self.cache_imgs_path}")
            self.load_cache()
        else:
            logger.info("Generating cache for test data...")
            self.load_data()
            self.save_cache()

    def load_data(self):
        _ims_all = (5 + 14 + 100 + 100 + 109 + 100) * 2
        for dataset in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K']:
            if dataset=='DIV2K':
                path=os.path.join(self.path,'../')
            else:
                path=self.path
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            if dataset=='DIV2K':
                files=files[:100]
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, 'HR', files[i])))
                im_hr = modcrop(im_hr, self.scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4]
                self.ims[key] = im_hr

                if dataset!='DIV2K':
                    lr_path='LR_bicubic/X%d'
                else:
                    lr_path='LR/X%d'

                im_lr = np.array(Image.open(
                    os.path.join(path, dataset, lr_path % self.scale, files[i][:-4] + 'x%d.png'%self.scale)))  # [:-4] + 'x%d.png'%scale)))
                if len(im_lr.shape) == 2:
                    im_lr = np.expand_dims(im_lr, axis=2)

                    im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                key = dataset + '_' + files[i][:-4] + 'x%d' % self.scale
                self.ims[key] = im_lr

                assert (im_lr.shape[0] * self.scale == im_hr.shape[0])

                assert (im_lr.shape[1] * self.scale == im_hr.shape[1])
                assert (im_lr.shape[2] == im_hr.shape[2] == 3)

        assert (len(self.ims.keys()) == _ims_all)

    def save_cache(self):
        np.save(self.cache_files_path, self.files, allow_pickle=True)
        np.save(self.cache_imgs_path, self.ims, allow_pickle=True)

    def load_cache(self):
        self.files=np.load(self.cache_files_path, allow_pickle=True).item()
        self.ims=np.load(self.cache_imgs_path, allow_pickle=True).item()


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size, debug=False, gpuNum=1, data_class=DIV2K, length=int(sys.maxsize)):
        self.data = data_class(scale, path, patch_size,debug=debug,gpuNum=gpuNum)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug=debug
        self.length=length

        self.is_cuda = True
        self.build()
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.length

    def build(self):
        self.data_iter = iter(DataLoader(
            dataset=self.data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        ))

    def next(self):
        try:
            # batch = self.data_iter.next()
            batch = next(self.data_iter)
            self.iteration += 1
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            return batch[0], batch[1]


class TestDataset(Dataset):
    @classmethod
    def get_init(self, dataset: str, srbm: SRBenchmark):
        def get(scale, path, patch_size, debug, gpuNum):
            assert srbm.scale==scale
            return TestDataset(dataset, srbm)
        return get
    def __init__(self, dataset:str, srbm:SRBenchmark):
        self.dataset=dataset
        self.srbm=srbm
        self.imgs=self.srbm.ims
    def __getitem__(self, index):
        files=self.srbm.files[self.dataset]
        filename=files[index%len(files)]
        hr_key=f"{self.dataset}_{filename[:-4]}"
        lr_key=hr_key+f"x{self.srbm.scale}"
        hr, lr=self.imgs[hr_key], self.imgs[lr_key]
        return lr, hr
    def __len__(self):
        return len(self.imgs)

class DebugDataset(Dataset):
    def __init__(self, num_image=2, num_channel=1, height=5, width=5, scale=4):
        self.num_image=num_image
        self.low_res=self.gen_img(width, height).repeat([num_image, num_channel, 1, 1])
        self.high_res=self.gen_img(scale*width, scale*height).repeat([num_image, num_channel, 1, 1])
        assert self.low_res.shape==(num_image, num_channel, height, width)
        assert self.high_res.shape==(num_image, num_channel, scale*height, scale*width)

    @staticmethod
    def gen_img(height, width):
        x=torch.linspace(0,1,width)
        y=torch.linspace(0,1,height)
        x_grid, y_grid=torch.meshgrid(x,y)
        img=(x_grid+y_grid)/2
        assert img.shape==(height,width)
        return img

    def __getitem__(self, index):
        return self.low_res[index], self.high_res[index]

    def __len__(self):
        return self.num_image

class DebugDataProvider:
    def __init__(self, dataset):
        self.dataset=dataset
        self.build()
        self.iteration=0
        self.epoch=0

    def build(self):
        self.data_iter=iter(DataLoader(dataset=self.dataset))

    def next(self):
        try:
            batch=next(self.data_iter)
            self.iteration+=1
            return batch
        except StopIteration:
            self.epoch+=1
            self.build()
            self.iteration+=1
            batch=next(self.data_iter)
            return batch
            
            
