from app.services.image_optimizer.background_removal_model import GetBackgroundRemovedByU2NET
from app.services.image_optimizer.bounding_boxes import BoundingBoxDetectionByFastRcnn
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import io

logger = logging.getLogger(__name__)

class RemoveBackground(object):
    
    def __init__(self):
        pass
    
    def get_coloured_background(self, input_image,background_colour):
        try:
            final_image = None
            if len(background_colour) == 4:
                final_image = Image.new('RGBA', input_image.size,(0,0,0,0))
                final_image.paste(input_image,(0,0),input_image)
                return final_image
            else:
                final_image = Image.new("RGB", input_image.size,background_colour)
                final_image.paste(input_image,(0,0),input_image)
            return final_image
        except Exception as e:
            logger.error(e,exc_info=True)
        
    def remove_background(self, image,background_colour):
        try:
            model = GetBackgroundRemovedByU2NET
            preprocessing_method = BoundingBoxDetectionByFastRcnn            
            final_image = model.process_image(image, preprocessing_method)  
            colour_background_image = self.get_coloured_background(final_image,background_colour)
            if len(background_colour) == 4:
                return colour_background_image
            image_array = Image.fromarray(np.array(colour_background_image))
            return image_array
        except Exception as e:
            logger.error(e,exc_info=True)
        
RemoveBackground = RemoveBackground()
