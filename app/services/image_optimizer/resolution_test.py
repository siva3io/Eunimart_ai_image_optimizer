from decimal import Decimal
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs_multistep
from tqdm import tqdm
import os
import numpy as np

class Trainer():

    def __init__(self, args, loader, my_model, my_loss, ckp):

        self.args = args
        self.scale=[4]
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer(args, self.model)
        self.error_last = 1e8


    def test(self):

        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.model.eval()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for low_resolution, high_resolution, filename in tqdm(d, ncols=80):
                    low_resolution, high_resolution = self.prepare(low_resolution, high_resolution)
                    super_resolution = self.model(low_resolution, idx_scale)
                    super_resolution = quantize(super_resolution,255)

                    save_list = [super_resolution]
                    image=self.ckp.save_results(d, filename[0], save_list, scale)

        torch.set_grad_enabled(True)
        return np.array(image)

    def prepare(self, *args):

        device = torch.device('cpu')
        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self,image):
        image=self.test()
        return image

class checkpoint():

    def __init__(self, args):

        self.args = args
        self.ok = True
        self.log = torch.Tensor()


    def save_results(self, dataset, filename, save_list, scale):

        if True:
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255/255)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        return np.array(tensor_cpu)
           

def quantize(img, rgb_range):

    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr':0.0001, 'weight_decay': 0}
    optimizer_class = optim.Adam
    kwargs_optimizer['betas'] =(0.9, 0.999)
    kwargs_optimizer['eps'] = 1e-08
    milestones = list(map(lambda x: int(x),['200']))
    kwargs_scheduler = {'milestones': milestones, 'gamma':0.5}
    scheduler_class = lrs_multistep.MultiStepLR

    class CustomOptimizer(optimizer_class):
        
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

