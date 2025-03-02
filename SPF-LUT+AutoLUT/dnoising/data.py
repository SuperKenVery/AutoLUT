import os
import threading
import time
import sys
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Provider(object):
    def __init__(self, batch_size, num_workers, sigma, path, patch_size):
        self.data = DIV2K(sigma, path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]

class DIV2K(Dataset):
    def __init__(self, sigma, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.sigma = sigma
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 901)]  # use both train and valid
        
        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f+".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]

        shape = lb.shape
        i = random.randint(0, shape[0]-self.sz)
        j = random.randint(0, shape[1]-self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i:i+self.sz, j:j+self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)

        lb = np.expand_dims(lb.astype(np.float32)/255.0, axis=0)

        # add Gaussian Noise
        im = lb + np.random.normal(0, self.sigma/255.0, lb.shape).astype(np.float32)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)
        

class BSD400(Dataset):
    def __init__(self, sigma, path, patch_size, rigid_aug=True):
        super(BSD400, self).__init__()
        self.sigma = sigma
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = ["test_" + str(i).zfill(3)
                          for i in range(1, 401)]  # "test_00X.png"

        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f+".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]

        shape = lb.shape
        i = random.randint(0, shape[0]-self.sz)
        j = random.randint(0, shape[1]-self.sz)

        lb = lb[i:i+self.sz, j:j+self.sz]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)

        lb = np.expand_dims(lb.astype(np.float32)/255.0, axis=0)

        # add Gaussian Noise
        im = lb + np.random.normal(0, self.sigma/255.0, lb.shape).astype(np.float32)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class DNBenchmark(Dataset):
    def __init__(self, path, sigma=5):
        super(DNBenchmark, self).__init__()
        datasets = ['Set12', 'BSD68', 'CBSD68', 'Kodak24', 'McMaster']
        self.ims = dict()
        self.files = dict()
        _ims_all = (12 + 68 + 68 + 24 + 18) * 2

        for dataset in datasets:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, files[i])))
                if dataset in ["Set12", "BSD68"]:
                    im_hr = im_hr[:, :, np.newaxis]

                key = dataset + '_' + files[i][:-4]

                self.ims[key] = im_hr

                np.random.seed(seed=0)  # set seed for random noise
                im_lr = im_hr / 255.0 + \
                    np.random.normal(0, sigma/255.0, im_hr.shape)

                key = dataset + '_' + files[i][:-4] + 'x%d' % sigma
                self.ims[key] = im_lr.astype(np.float32)

                # assert (im_lr.shape[2] == im_hr.shape[2] == 1)

        assert (len(self.ims.keys()) == _ims_all)
