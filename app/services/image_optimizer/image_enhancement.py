import cv2
import numpy as np
from .check_Image_quality_score import ImageQualityValue,BlurOrNot,BrightnessAndContrastScore,CheckResolutionOfImage
from .adjust_brightness_and_contrast_and_enhance_image import BrightnessAndContrastEnhancement
from .blur_removal import RemoveBlur
from PIL import Image
from io import BytesIO
import uuid
import os
import io
import requests
import base64
from app.utils import upload_to_s3,download_from_s3
from PIL import Image, ImageEnhance
from .resolution_main import ResolutionEnhancement

class ImageEnhancementCommon(object):

    def image_blur_removal(self,image,is_human):
        check_blur_image = True
        is_human = True

        ### @updated: 16 June 2022 - commenting the blur score > 80 condition. to enable uncomment the below code
        #check_blur_image = BlurOrNot.check_blur(image)

        if check_blur_image:
            if is_human:
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = RemoveBlur.remove_blur(image)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = ImageEnhance.Sharpness(image).enhance(1.2)
        
        return check_blur_image,image

    def image_resolution_enhancement(self,image,is_human):
        ResolutionEnhancements = ResolutionEnhancement()
        is_resolution_enhancement_required = CheckResolutionOfImage.check_resolution(image)
        if is_resolution_enhancement_required:

            if is_human:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = ResolutionEnhancements.main(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = ResolutionEnhancements.main(image)
                image = Image.fromarray(image)
        return image

    def image_brightness_contrast_and_color_enhancement(self,image,is_human):

        brightness_score = BrightnessAndContrastScore.brightness_score(image)
        contrast_score = BrightnessAndContrastScore.contrast_quality(image)
        image_quality_score = ImageQualityValue.test_measure_BRISQUE(image)

        if is_human:

            if image_quality_score >= 80 or brightness_score <= 120 or contrast_score <= 2:
            
                image = BrightnessAndContrastEnhancement.adjust_gamma(image)     
                image = BrightnessAndContrastEnhancement.automatic_brightness_and_contrast(image)
        else:

            if image_quality_score >= 45 or brightness_score <= 120 or contrast_score <= 2:
                image = ImageEnhance.Color(image).enhance(1.2)
                image = ImageEnhance.Brightness(image).enhance(1.2)
                image = ImageEnhance.Contrast(image).enhance(1.2)

        return image

    def send_file_path(self,request_data,input_image,output_image,is_human):        

        if is_human:
            output_image_byte_array = cv2.imencode('.'+request_data["image_file_format"], output_image)[1].tostring()      
        else:
            try:
                output_image_byte_array = io.BytesIO()
                output_image.save(output_image_byte_array, format=request_data["image_file_format"])               
                output_image_byte_array = output_image_byte_array.getvalue()
            except:
                output_image_byte_array = cv2.imencode('.'+request_data["image_file_format"], output_image)[1].tostring()
        input_image_byte_array = io.BytesIO()              
        input_image_byte_array = input_image_byte_array.getvalue()
        file_name = str(uuid.uuid4()) + '.' + request_data["image_file_format"]
        input_file_path  = upload_to_s3(input_image_byte_array,request_data, 'input/'+file_name)
        output_file_path = upload_to_s3(output_image_byte_array,request_data, 'output/'+file_name)
        return output_file_path,base64.b64encode(output_image_byte_array)


EnhancedImage=ImageEnhancementCommon()

        



        
