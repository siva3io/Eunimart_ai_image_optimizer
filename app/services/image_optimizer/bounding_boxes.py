import logging
import numpy as np
from mxnet import nd
from PIL import Image
from gluoncv import model_zoo, data
logger = logging.getLogger(__name__)

__net__ = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
class_names = __net__.classes  

class BoundingBoxDetectionByFastRcnn(object):
    
    
    def __init__(self):
        self.model = None
        self.prep_image = None
        self.orig_image = None
        pass

    def trans_paste(self, bg_img, fg_img, box=(0, 0)):
        '''
        Inserts an image into another image while maintaining transparency.
        '''
        try:
            fg_img_trans = Image.new("RGBA", bg_img.size)
            fg_img_trans.paste(fg_img, box, mask=fg_img)
            new_img = Image.alpha_composite(bg_img, fg_img_trans)
            return new_img
        except Exception as e:
            logger.error(e,exc_info=True)

    def orig_object_border(self, border, orig_image = None, resized_image = None, indent=16):
        '''
        Rescale the bounding boxes of an object
        '''
        try:
            x_factor = resized_image.shape[1] / orig_image.size[0] # for x-axis coordinates
            y_factor = resized_image.shape[0] / orig_image.size[1] #for y-axis coordinates
            xmin, ymin, xmax, ymax = [int(x) for x in border]
            if ymin < 0:
                ymin = 0
            if ymax > resized_image.shape[0]:
                ymax = resized_image.shape[0]
            if xmax > resized_image.shape[1]:
                xmax = resized_image.shape[1]
            if xmin < 0:
                xmin = 0
            if x_factor == 0:
                x_factor = 1
            if y_factor == 0:
                y_factor = 1
            border = (int(xmin / x_factor) - indent,
                      int(ymin / y_factor) - indent, int(xmax / x_factor) + indent, int(ymax / y_factor) + indent)
            return border
        except Exception as e:
            logger.error(e,exc_info=True)
        
    def load_image(self, data_input):
        '''
        load and preprocess the image
        '''
        try:
            if isinstance(data_input, str):
                data_input = Image.open(data_input)
                data_input = data_input.convert("RGB")
                image = np.array(data_input)  # Convert PIL image to numpy arr
            else:
                data_input = data_input.convert("RGB")
                image = np.array(data_input)  
                
            input_mxnet_ndarray, resized_image = data.transforms.presets.rcnn.transform_test(nd.array(image))
            return input_mxnet_ndarray, resized_image
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def process_image(self, image):
        '''
        get resized images and get the ids and 
        scores of each box(object) of the image
        '''
        try:
            input_mxnet_ndarray, resized_image = self.load_image(image)
            ids, scores, bboxes = [each_input[0].asnumpy() for each_input in __net__(input_mxnet_ndarray)]
            return resized_image, {"ids": ids, "scores": scores, "bboxes": bboxes}  
        except Exception as e:
            logger.error(e,exc_info=True) 
    
    def get_objects_from_image(self, model, prepared_image, orig_image):
        '''
        find out the total objects in the image
        '''
        try:
            resized_image, results = self.process_image(orig_image)
            classes = class_names
            bboxes, ids, scores = results['bboxes'], results['ids'], results['scores']
            if (len(bboxes) < 1) or (ids is not None and not len(bboxes) == len(ids)) or (scores is not None and not len(bboxes) == len(scores)): 
                return model.get_output(prepared_image, orig_image) #here model is the instance of GetBackgroundRemovedByU2NET class
            objects = []
            for box_index, bbox in enumerate(bboxes):
                if (scores is not None and scores.flat[box_index] < 0.5) or (ids is not None and ids.flat[box_index] < 0):
                    continue
                object_cls_id = int(ids.flat[box_index]) if ids is not None else -1
                if classes is not None and object_cls_id < len(classes):
                    object_label = classes[object_cls_id]
                else:
                    object_label = str(object_cls_id) if object_cls_id >= 0 else ''
                object_border = self.orig_object_border(bbox, orig_image, resized_image)
                objects.append([object_label, object_border])
            return objects
        except Exception as e:
            logger.error(e,exc_info=True)
    
    def get_preprocessed_image(self, model, prepared_image, orig_image):
        '''
        take each object and pass it through the get_output function
        '''
        try:
            objects = self.get_objects_from_image(model, prepared_image, orig_image)    
            if objects:
                if len(objects) == 1:
                    return model.get_output(prepared_image, orig_image)
                    
                else:
                    obj_images = []
                    for obj in objects:
                        border = obj[1]
                        obj_crop = orig_image.crop(border)
                        obj_img = model.process_image(obj_crop) #(if obj[0]=="person" or not, same codeline)
                        obj_images.append([obj_img, obj])
                    image = Image.new("RGBA", orig_image.size)
                    for obj in obj_images:
                        image = self.trans_paste(image, obj[0], obj[1][1])
                    return image
            else:
                return model.get_output(prepared_image, orig_image)
        except Exception as e:
            logger.error(e,exc_info=True)
    
BoundingBoxDetectionByFastRcnn = BoundingBoxDetectionByFastRcnn() 