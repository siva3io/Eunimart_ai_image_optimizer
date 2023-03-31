import cv2
import numpy as np
from PIL import Image
from io import BytesIO

class AdjustBrightnessAndContrastAndEnhanceImage(object):
    
    '''This class is for automatic adjusting Brightness and Contrast '''

    # I found the value of `gamma=1.4` to be the best in our case
    def adjust_gamma(self,image, gamma=1.4):
        
        '''This funtion adjust gamma in the input image and add non linearity to the function'''
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        image = np.array(image)
        inv_gamma = 1.0 / gamma
        table = np.array([((pixels / 255.0) ** inv_gamma) * 255
            for pixels in np.arange(0, 256)]).astype("uint8")
    
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def convert_scale(self,img, alpha, beta):
        
        '''This function apply the transform newimage=img*aplha+beta'''
        
        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)
    
    def automatic_brightness_and_contrast(self,image,clip_hist_percent=25):
        
        '''Automatic brightness and contrast optimization with optional histogram clipping'''
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
    
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
    
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
    
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
    
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
    
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
    
        auto_enhanced_result = self.convert_scale(image, alpha=alpha, beta=beta)
        return auto_enhanced_result

BrightnessAndContrastEnhancement=AdjustBrightnessAndContrastAndEnhanceImage()
