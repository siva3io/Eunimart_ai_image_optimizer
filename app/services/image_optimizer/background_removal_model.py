import os
import logging
import torch
import numpy as np
import requests
from PIL import Image
from skimage import io, transform
from io import BytesIO
from app.utils import download_from_s3
logger = logging.getLogger(__name__)
from torch.autograd import Variable
from app.services.image_optimizer.backround_removal_model_architecture import U2NET as U2NET_DEEP

logger = logging.getLogger(__name__)

net = U2NET_DEEP()
abs_model_path = "app/services/image_optimizer/model/"
model_path = "u2net.pth"
s3_path = "background_removal/"
if not os.path.exists(abs_model_path + model_path):
    download_from_s3(s3_path + model_path,abs_model_path + model_path)

if torch.cuda.is_available():
    net.load_state_dict(torch.load(abs_model_path + model_path))
    net.cuda()
else:
    net.load_state_dict(torch.load(abs_model_path + model_path, map_location=torch.device("cpu")))
    net.eval()

class GetBackgroundRemovedByU2NET(object):

    def __init__(self):
        pass
    
    def transform_ndrarray_to_tensor(self, image: np.ndarray):
        '''
        transform the nd_array to tensor
        '''
        try:
            tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
            image /= np.max(image)
            if image.shape[2] == 1:
                tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            tmp_img = tmp_img.transpose((2, 0, 1))
            tmp_img = np.expand_dims(tmp_img, 0)
            return torch.from_numpy(tmp_img)
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def load_image(self, image):
        '''
        loading and pre-processing the image
        '''
        try:
            image_size = 320  # Size of the input and output image for the model
            image = np.array(image)
            pil_image = Image.fromarray(image)
            image = transform.resize(image, (image_size, image_size), mode='constant') 
            image = self.transform_ndrarray_to_tensor(image)  # Convert image from numpy arr to tensor
            return image, pil_image
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def normalize(self, predicted):
        '''
        normalizing the mask
        '''
        try:
            maxim = torch.max(predicted)
            minim = torch.min(predicted)
            out = (predicted - minim) / (maxim - minim)
            return out
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def prepare_mask(self, predict, image_size):
        '''
        preparing the  mask
        '''
        try:
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()
            mask = Image.fromarray(predict_np * 255).convert("L")
            mask = mask.resize(image_size, resample=Image.BILINEAR)
            return mask
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def get_output(self, image, orig_image):
        '''
        this method results in creating the 
        alpha image of a given image which wil be final output
        '''
        try:
            image = image.type(torch.FloatTensor)
            if torch.cuda.is_available():
                image = Variable(image.cuda())
            else:
                image = Variable(image)        
            mask, d2, d3, d4, d5, d6, d7 = net(image) 
            mask = mask[:, 0, :, :] #normalizing
            mask = self.normalize(mask)
            mask = self.prepare_mask(mask, orig_image.size) 
            empty = Image.new("RGBA", orig_image.size, (255,255,255,0))
            image = Image.composite(orig_image, empty, mask) # Apply mask to image
            return image
        except Exception as e:
            logger.error(e,exc_info=True)

    def process_image(self, image, preprocessing=None):
        '''
        for a given image, here we load, preprocess,
        and get the final_image_output
        '''
        try:
            image, orig_image = self.load_image(image)  # Load image
            if image is False or orig_image is False:
                return False
            if preprocessing:  
                image = preprocessing.get_preprocessed_image(self, image, orig_image)  #self in the arg is the class instance
            else:
                image = self.get_output(image, orig_image)  
            return image
        except Exception as e:
            logger.error(e,exc_info=True)
            
GetBackgroundRemovedByU2NET = GetBackgroundRemovedByU2NET()
    
