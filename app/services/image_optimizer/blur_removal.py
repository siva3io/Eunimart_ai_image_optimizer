from .blur_removal_model import RemoveBlurModel

class BlurRemoval(object):
    
    '''This class is the main Blur removal class'''
    
    def remove_blur(self,image):
        
        image=RemoveBlurModel.blur_removal_function(image)
        
        return image

RemoveBlur=BlurRemoval()
