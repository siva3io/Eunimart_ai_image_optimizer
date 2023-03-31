import os
import torch
import torch.utils.data as data
from torch.utils.data import dataloader
import random
import numpy as np
import skimage.color as skicolor

def set_channel(*args, n_channels=3):
   
    def _set_channel(img):

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        channels = img.shape[2]
        if n_channels == 1 and channels == 3:
            img = np.expand_dims(skicolor.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and channels == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img
    return [_set_channel(image) for image in args]

def np2Tensor(*args, rgb_range=255):

    def _np2Tensor(img):

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(image) for image in args]



class Data(object):
    
    def __init__(self, args,image):
        
        self.loader_train = None
        self.loader_test = []
        types='Demo'
        testset=DataInput(None,image,train=False, name=types)
        self.loader_test.append(dataloader.DataLoader(testset,batch_size=1,shuffle=False,pin_memory=not False,num_workers=0,))

            

class DataInput(data.Dataset):

    def __init__(self, args,image,name='Demo', train=False, benchmark=False):

        self.args = args
        self.name = name
        self.scale=[4]
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark
        self.filelist = []
        self.filelist.append(np.array(image))
        
    def __getitem__(self, idx):

        low_resolution=self.filelist[0]
        low_resolution, = set_channel(low_resolution, n_channels=3)
        low_resolution_tensor, =np2Tensor(low_resolution, rgb_range=255)
        return low_resolution_tensor,-1,"image"

    def __len__(self):

        return len(self.filelist)

    def set_scale(self, idx_scale):

        self.idx_scale = idx_scale
