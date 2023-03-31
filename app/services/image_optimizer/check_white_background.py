import sys                                  
import cv2                                         
import numpy as np
from io import BytesIO
from PIL import Image

class CheckBackground(object):
    
    '''This class implements checking background white or not'''
    
    def __init__(self):
        pass
    
    def count_colors(self, image):
        
        '''This funcion count the number of color'''
        
        colors_count = {}
        (channel_b, channel_g, channel_r) = cv2.split(image)
        channel_b = channel_b.flatten()
        channel_g = channel_g.flatten() 
        channel_r = channel_r.flatten() 
        for pixels in range(len(channel_b)):
            RGB = "(" + str(channel_r[pixels]) + "," + \
                str(channel_g[pixels]) + "," + str(channel_b[pixels]) + ")"
            if RGB in colors_count:
                colors_count[RGB] += 1
            else:
                colors_count[RGB] = 1
        return colors_count

    def check_most_frequent_colour(self, colors_count):
        
        '''This funtion count the most frequent colour in dictionary'''
        
        sort_keys=sorted(colors_count, key=colors_count.__getitem__)
        if sort_keys[-1]=='(255,255,255)':
            return True
        else:
            return False

    def ten_most_frequent_colour(self, colors_count):
        
        '''This funtion count the most frequent colour in dictionary'''
        
        sort_keys=sorted(colors_count, key=colors_count.__getitem__)
        ten_colours=sort_keys[len(sort_keys)-20:len(sort_keys)]
        return ten_colours
          
    def check_white_or_not(self,image):
        
        '''This funcion check white or not '''
        
        image = np.array(image)
        colors_count = self.count_colors(image)
        white_or_not = self.check_most_frequent_colour(colors_count)
        return white_or_not

    def find_most_frequent_colour(self, image):
        
        '''This funcion check white or not '''
        
        image = np.array(image)
        colors_count = self.count_colors(image)
        colors = self.ten_most_frequent_colour(colors_count)
        return colors
      
CheckBackground = CheckBackground()