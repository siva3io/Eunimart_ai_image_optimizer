import os
import random
import scipy.misc
import numpy as np

from app.utils import download_from_s3
from PIL import Image
import cv2
from .models import CNN
import torch
import tempfile

from torchvision.transforms import transforms
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
abs_model_path = "app/services/image_optimizer/model/"
model_path = "checkpoints/color/"
s3_model_path = "blur_model/checkpoints/color/"
blur_removal_model_path = "new_blur_model.pth"

if not os.path.exists(abs_model_path + model_path):
    os.makedirs(abs_model_path + model_path) 
    for checkpoint in ['deblur.model-523000.data-00000-of-00001','deblur.model-523000.index','deblur.model-523000.meta', blur_removal_model_path]:
        download_from_s3(s3_model_path + checkpoint,abs_model_path + model_path + checkpoint)

class BlurRemovalModel(object):

    def init(self) -> None:
        pass


    # @classmethod
    def blur_removal_function(self, img):
        device = 'cpu'

        # load the trained model
        model = CNN().to(device).eval()
        model.load_state_dict(torch.load(abs_model_path + model_path + blur_removal_model_path,
                                         map_location=torch.device('cpu')))

        # define transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image = img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = transform(image).unsqueeze(0)
        print(image.shape)

        with torch.no_grad():
            outputs = model(image)
            
        with tempfile.NamedTemporaryFile(suffix='.jpg') as t:
            save_image(outputs.cpu().data,t.name)
            img = Image.open(t.name)
        return img

RemoveBlurModel = BlurRemovalModel()
