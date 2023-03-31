import os
import logging
import numpy as np
import requests
import cv2
import urllib
from .get_colors import Colors
from .image_enhancement import EnhancedImage
from .background_removal import RemoveBackground
from .check_white_background import CheckBackground
from app.utils import catch_exceptions
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
import base64
import json
from config import Config

logger = logging.getLogger(name=__name__)

class Optimizer(object):

    def __init__(self):
        pass
    
    @catch_exceptions
    def load_image(self,image_url):
        try:
            parsed_image = urlparse(image_url)
            if all([parsed_image.scheme, parsed_image.netloc, parsed_image.path]):
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
            else:
                bytes_string = base64.b64decode(image_url)
                image = Image.open(BytesIO(bytes_string))
            if image.mode in ("RGBA", "P"):     #https://stackoverflow.com/questions/48206051/image-conversion-cannot-write-mode-rgba-as-jpeg
                image = image.convert("RGB")
            return image
        except Exception as e:
            logger.error(e,exc_info=True)
            return ''
    
    @catch_exceptions
    def check_is_human(self,request_data):
        try:
            is_human = False
            get_hierarchy_obj = json.loads(requests.post(url = Config.GET_HIERARCHY_ENDPOINT,json= request_data).text)
            if get_hierarchy_obj["status"]:
                if get_hierarchy_obj["data"]["category_name"] == "Fashion":
                    is_human = True
            return is_human
        except Exception as e:
            return is_human

   
    @catch_exceptions
    def optimize(self,request_data):
        try:
            response_data = {}
            mandatory_fields = ["sku_id","account_id","channel_id","image","selected_parameters"]
            for field in mandatory_fields:
                if request_data.get("data") and not field in request_data["data"]:
                    response_data = {
                        "status":False,
                        "message":"Required field is missing",
                        "error_obj":{
                            "description":"{} is missing".format(field),
                            "error_code":"REQUIRED_FIELD_IS_MISSING"
                        }
                    }
            if not response_data:
                if "image_file_format" not in request_data["data"] or not request_data["data"]["image_file_format"]: 
                    request_data["data"]["image_file_format"] = "jpeg"          #changed default format as jpeg
                    if request_data["data"]["background_colour"]=='0,0,0,0':    #png format for transparent background
                        request_data["data"]["image_file_format"] = "png"       #code updated on 10/12/2021

                image_url = request_data["data"]["image"]
                rgb_list = []
                for color in request_data["data"]["background_colour"].split(','):
                    rgb_list.append(int(color))
                background_colour = tuple(rgb_list)
                image = self.load_image(image_url)
                if image:
                    response_data = {
                        "status":True,
                        "message": "Successfully optimised the image",
                        "data" : {
                            "color" : Colors.get_colors(image),
                            "is_white_background" : CheckBackground.check_white_or_not(image)                        
                            },
                        "columns":[
                            {
                                "column_name"     : "Color",
                                "column_key"      : "color",
                                "column_position" : 1
                            },
                            {
                                "column_name"     : "Is White Background",
                                "column_key"      : "is_white_background",
                                "column_position" : 2
                            }
                            
                        ]
                    }

                    input_image = image
                    is_human = self.check_is_human(request_data)

                    if "background_removal" in request_data["data"]["selected_parameters"]:
                        image = RemoveBackground.remove_background(image,background_colour)
                        if len(background_colour) == 3:
                            color_image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
                            white_background_image_url,white_background_image_bytes = EnhancedImage.send_file_path(request_data["data"],input_image,color_image,True)
                        else:
                            white_background_image_url,white_background_image_bytes = EnhancedImage.send_file_path(request_data["data"],input_image,image,False)
                        response_data["data"]["white_background_image"] = white_background_image_url
                        response_data["data"]["white_background_image_bytes"] = str(white_background_image_bytes)
                        response_data["columns"].append({"column_name":"White Background Image", "column_key":"white_background_image", "column_position":4})

                    if "quality_enhancement" in request_data["data"]["selected_parameters"]:
                        image = EnhancedImage.image_brightness_contrast_and_color_enhancement(image,is_human)
                    
                    if is_human:
                        image = np.array(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if "blur_removal" in request_data["data"]["selected_parameters"]:
                        is_blur,image = EnhancedImage.image_blur_removal(image,is_human)
                        response_data["data"]["is_blur"] = is_blur
                        response_data["columns"].append({"column_name":"Is Blur", "column_key":"is_blur", "column_position":5})

                    
                    if "resolution_enhancement" in request_data["data"]["selected_parameters"]:
                        image = EnhancedImage.image_resolution_enhancement(image,is_human)

                    # response_data["data"]["enhanced_image"] 
                    signed_image_url,image_bytes = EnhancedImage.send_file_path(request_data["data"],input_image,image,is_human)
                    response_data["data"]["enhanced_image"] = signed_image_url
                    response_data["data"]["enhanced_image_bytes"] = str(image_bytes)
                    response_data["columns"].append({"column_name":"Enhanced Image", "column_key":"enhanced_image", "column_position":3})

                else:
                    response_data = {
                            "status" : False,
                            "message" : "Image url is not valid",
                            "error_obj" : {
                                "description":"image_url is not valid",
                                "error_code":"INVALID_FIELD_IS_GIVEN"
                            }
                        }
            return response_data

        except Exception as e:
            logger.error(e,exc_info=True)
            return ''

                
ImageOptimizer_new = Optimizer()
