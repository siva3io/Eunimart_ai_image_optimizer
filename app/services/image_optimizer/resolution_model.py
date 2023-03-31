import os
import torch
import torch.nn as nn
import torch.utils.model_zoo
import torch.nn.functional as F
import math


class Model(nn.Module):

    def __init__(self, args, ckp):

        super(Model, self).__init__()
        self.scale = [4]
        self.idx_scale = 0
        self.self_ensemble = False
        self.chop = False
        self.precision = 'single'
        self.cpu = True
        self.n_GPUs = 1
        self.save_models = False
        self.model = self.make_model(None).to('cpu')
        self.model_path='/home/ubuntu/vdezi_ai_image_optimizer/app/services/models'
        self.load(self.model_path,pre_train='download',resume=0,cpu=True)
        
    def make_model(self,args, parent=False):

        return EDSR(args)
    
    def forward(self, inputs, idx_scale):

        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, inputs, range(self.n_GPUs))
            else:
                return self.model(inputs)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(inputs, forward_function=forward_function)
            else:
                return forward_function(inputs)

    def load(self, apath, pre_train='', resume=-1, cpu=False):

        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        load_from = torch.load(os.path.join(apath, 'edsr_baseline_x4-6b446fab.pt'),**kwargs)

        if load_from:
            self.model.load_state_dict(load_from, strict=False)
   

class MeanShift(nn.Conv2d):


    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for param in self.parameters():
            param.requires_grad = False

class BasicBlock(nn.Sequential):


    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        batch_normslization=True, act=nn.ReLU(True)):
        
        model_basic_block = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if batch_normslization:
            model_basic_block.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            model_basic_block.append(act)

        super(BasicBlock, self).__init__(*model_basic_block)

class ResBlock(nn.Module):


    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, batch_normaliation=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        model_basic_block_ = []
        for block in range(2):
            model_basic_block_.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if batch_normaliation:
                model_basic_block_.append(nn.BatchNorm2d(n_feats))
            if block == 0:
                model_basic_block_.append(act)

        self.body = nn.Sequential(*model_basic_block_)
        self.res_scale = res_scale

    def forward(self, inputs):
        result = self.body(inputs).mul(self.res_scale)
        result += inputs

        return result

class Upsampler(nn.Sequential):


    def __init__(self, conv, scale, n_feats, batch_normaliation=False, act=False, bias=True):

        model_up = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                model_up.append(conv(n_feats, 4 * n_feats, 3, bias))
                model_up.append(nn.PixelShuffle(2))
                if batch_normaliation:
                    model_up.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    model_up.append(nn.ReLU(True))
                elif act == 'prelu':
                    model_up.append(nn.PReLU(n_feats))

        elif scale == 3:
            model_up.append(conv(n_feats, 9 * n_feats, 3, bias))
            model_up.append(nn.PixelShuffle(3))
            if batch_normaliation:
                model_up.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                model_up.append(nn.ReLU(True))
            elif act == 'prelu':
                model_up.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*model_up)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):

    
    def __init__(self, args):

        conv=self.default_conv
        super(EDSR, self).__init__()
        n_resblocks=16
        n_feats=64
        kernel_size = 3 
        scale=4
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        model_head = [conv(3, n_feats, kernel_size)]

        # define body module
        model_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        model_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        model_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats,3, kernel_size)
        ]

        self.head = nn.Sequential(*model_head)
        self.body = nn.Sequential(*model_body)
        self.tail = nn.Sequential(*model_tail)

    def forward(self, x_input):

        x_input = self.sub_mean(x_input)
        x_input = self.head(x_input)

        result = self.body(x_input)
        result += x_input

        x_input = self.tail(result)
        x_input = self.add_mean(x_input)

        return x_input

    def load_state_dict(self, state_dict, strict=True):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
  
    def default_conv(self,in_channels, out_channels, kernel_size, bias=True):
        
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


