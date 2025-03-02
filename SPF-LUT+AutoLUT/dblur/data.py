import os
import threading
import time
import sys
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Provider(object):
    def __init__(self, batch_size, num_workers, path, patch_size):
        self.data = GoPro(path, patch_size)
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
            batch = next(self.data_iter)
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


class GoPro(Dataset):
    """Basic dataloader class
    """

    def __init__(self, path, patch_size, rigid_aug=True):
        super(GoPro, self).__init__()
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path

        self.blur_list = []
        self.sharp_list = []

        self.set_keys()
        
        self._scan(path)
        
        print(path, len(self.blur_list))
        
        self.hr_cache = os.path.join(path, "cache_sharp.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_blur.npy")
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)
        exit()

    def cache_hr(self):
        hr_dict = dict()
        for f in self.sharp_list:
            hr_dict[f] = np.array(Image.open(f))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def cache_lr(self):
        lr_dict = dict()
        for f in self.blur_list:
            lr_dict[f] = np.array(Image.open(f))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)
        
    def set_keys(self):
        self.blur_key = 'blur'      # to be overwritten by child class
        self.sharp_key = 'sharp'    # to be overwritten by child class

        self.non_blur_keys = []
        self.non_sharp_keys = []

        return

    def _scan(self, root=None):
        """Should be called in the child class __init__() after super
        """
        if root is None:
            root = self.subset_root

        if self.blur_key in self.non_blur_keys:
            self.non_blur_keys.remove(self.blur_key)
        if self.sharp_key in self.non_sharp_keys:
            self.non_sharp_keys.remove(self.sharp_key)

        def _key_check(path, true_key, false_keys):
            path = os.path.join(path, '')
            if path.find(true_key) >= 0:
                for false_key in false_keys:
                    if path.find(false_key) >= 0:
                        return False

                return True
            else:
                return False

        def _get_list_by_key(root, true_key, false_keys):
            data_list = []
            for sub, dirs, files in os.walk(root):
                if not dirs:
                    file_list = [os.path.join(sub, f) for f in files]
                    if _key_check(sub, true_key, false_keys):
                        data_list += file_list

            data_list.sort()

            return data_list

        def _rectify_keys():
            self.blur_key = os.path.join(self.blur_key, '')
            self.non_blur_keys = [os.path.join(non_blur_key, '') for non_blur_key in self.non_blur_keys]
            self.sharp_key = os.path.join(self.sharp_key, '')
            self.non_sharp_keys = [os.path.join(non_sharp_key, '') for non_sharp_key in self.non_sharp_keys]

        _rectify_keys()

        self.blur_list = _get_list_by_key(root, self.blur_key, self.non_blur_keys)
        self.sharp_list = _get_list_by_key(root, self.sharp_key, self.non_sharp_keys)

        if len(self.sharp_list) > 0:
            assert(len(self.blur_list) == len(self.sharp_list))

        return
    
    def __getitem__(self, _dump):
        key = random.choice(range(len(self.blur_list)))
        # print(self.blur_list[key], self.sharp_list[key])
        im = self.lr_ims[self.blur_list[key]]
        lb = self.hr_ims[self.sharp_list[key]]

        shape = lb.shape
        i = random.randint(0, shape[0]-self.sz)
        j = random.randint(0, shape[1]-self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i:i+self.sz, j:j+self.sz, c]
        im = im[i:i+self.sz, j:j+self.sz, c]

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

        lb = np.expand_dims(lb.astype(np.float32)/255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32)/255.0, axis=0)

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

class GoProTest(Dataset):
    """Basic dataloader class
    """

    def __init__(self, path):
        super(GoProTest, self).__init__()
        self.path = path

        self.blur_list = []
        self.sharp_list = []

        self.set_keys()
        
        self._scan(path)
        
        print(path, len(self.blur_list))
        
        self.hr_cache = os.path.join(path, "cache_sharp.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_blur.npy")
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)

    def cache_hr(self):
        hr_dict = dict()
        for f in self.sharp_list:
            hr_dict[f] = np.array(Image.open(f))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def cache_lr(self):
        lr_dict = dict()
        for f in self.blur_list:
            lr_dict[f] = np.array(Image.open(f))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)
        
    def set_keys(self):
        self.blur_key = 'blur'      # to be overwritten by child class
        self.sharp_key = 'sharp'    # to be overwritten by child class

        self.non_blur_keys = []
        self.non_sharp_keys = []

        return

    def _scan(self, root=None):
        """Should be called in the child class __init__() after super
        """
        if root is None:
            root = self.subset_root

        if self.blur_key in self.non_blur_keys:
            self.non_blur_keys.remove(self.blur_key)
        if self.sharp_key in self.non_sharp_keys:
            self.non_sharp_keys.remove(self.sharp_key)

        def _key_check(path, true_key, false_keys):
            path = os.path.join(path, '')
            if path.find(true_key) >= 0:
                for false_key in false_keys:
                    if path.find(false_key) >= 0:
                        return False

                return True
            else:
                return False

        def _get_list_by_key(root, true_key, false_keys):
            data_list = []
            for sub, dirs, files in os.walk(root):
                if not dirs:
                    file_list = [os.path.join(sub, f) for f in files]
                    if _key_check(sub, true_key, false_keys):
                        data_list += file_list

            data_list.sort()

            return data_list

        def _rectify_keys():
            self.blur_key = os.path.join(self.blur_key, '')
            self.non_blur_keys = [os.path.join(non_blur_key, '') for non_blur_key in self.non_blur_keys]
            self.sharp_key = os.path.join(self.sharp_key, '')
            self.non_sharp_keys = [os.path.join(non_sharp_key, '') for non_sharp_key in self.non_sharp_keys]

        _rectify_keys()

        self.blur_list = _get_list_by_key(root, self.blur_key, self.non_blur_keys)
        self.sharp_list = _get_list_by_key(root, self.sharp_key, self.non_sharp_keys)

        if len(self.sharp_list) > 0:
            assert(len(self.blur_list) == len(self.sharp_list))

        return

    def __len__(self):
        return len(self.blur_list)
    

class DNBenchmark(Dataset):
    def __init__(self, path, sigma=5):
        super(DNBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (12 + 68) * 2

        for dataset in ['Set12', 'BSD68']:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, files[i])))
                im_hr = im_hr[:, :, np.newaxis]

                key = dataset + '_' + files[i][:-4]

                self.ims[key] = im_hr

                np.random.seed(seed=0)  # set seed for random noise
                im_lr = im_hr / 255.0 + \
                    np.random.normal(0, sigma/255.0, im_hr.shape)

                key = dataset + '_' + files[i][:-4] + 'x%d' % sigma
                self.ims[key] = im_lr.astype(np.float32)

                assert (im_lr.shape[2] == im_hr.shape[2] == 1)

        assert (len(self.ims.keys()) == _ims_all)
