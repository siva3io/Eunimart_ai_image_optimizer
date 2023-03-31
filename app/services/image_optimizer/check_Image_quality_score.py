import cv2
import numpy as np
import math as maths
from scipy.special import gamma as gammas_value
import os
import libsvm.svm
import libsvm.svmutil
from libsvm.svm import *
from libsvm.svmutil import *
from libsvm import svm
from libsvm import svmutil
from skimage import io
from skimage import exposure
from skimage import color
from skimage import img_as_float
import numpy as np
from sklearn.metrics import mutual_info_score as mutuals
from io import BytesIO
import math
from app.utils import download_from_s3
from PIL import Image, ImageStat

abs_model_path = "app/services/image_optimizer/model/"
model_path = "brisq"
s3_path = "image_quality_check/"


if not os.path.exists(abs_model_path + model_path):
    download_from_s3(s3_path + model_path,abs_model_path + model_path)

class CheckResolution(object):
    
    '''This class used for checking resolution of the image'''
    
    def check_resolution(self,image):
        
        image = np.array(image)
        width,height=image.shape[0],image.shape[1]
         
        if width<=500 and height<=500:
            return True
        
        return False
        

         
class CheckBlur(object):
    
    '''This class is used for checking blur in the image'''
    
    def variance_of_laplacian(self,image):
    	
    	return cv2.Laplacian(image, cv2.CV_64F).var()
    
    def check_blur(self,image):
        
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = self.variance_of_laplacian(gray)
        if blur_score > 80:
            return False
        return True

class ImageContrastAndBrightnessScore(object):
    
    '''This class implements functions which compute brightness and contrast score'''
    
    def brightness_score(self,image):
        
        '''This function implements the brightness score'''
        stat = ImageStat.Stat(image)
        red,green,blue = stat.mean[:3]
        return math.sqrt(0.241*(red**2) + 0.691*(green**2) + 0.068*(blue**2))
    
    
    def compute_contrast_quality_for_image(self,input_image, num_bins=128):
        
        '''This function implement the function which calculate the contrast level '''

        # Check dimensions of input image
        # If image dimensions is 2, then it is a gray-scale image
        # First convert input to RGB image
        input_image = np.array(input_image)
        if input_image.shape == 2:
            input_image = color.gray2rgb(input_image)
        if input_image.shape[2] == 4:
            input_image=color.rgba2rgb(input_image)
    
        # Convert the RGB image to HSV. Exposure is primarily correlated with Value rather
        # than Hue and Saturation
        image_hsv = color.rgb2hsv(input_image)
    
        # The intensity channel is third in HSV format image
        v_channel = image_hsv[:, :, 2]
    
        # compute the contrast equalized array of intensity channel of image
        v_channel_equalized = exposure.equalize_hist(v_channel, nbins=num_bins)
    
        # compute the histogram of intensity channel
        v_channel_histogram, histogram_bin_edges = np.histogram(img_as_float(v_channel), bins=num_bins, density=True)
    
        # compute the histogram of contrast equalized intensity channel
        v_channel_equalized_histogram, _ = np.histogram(img_as_float(v_channel_equalized), bins=num_bins, density=True, range=(histogram_bin_edges[0], histogram_bin_edges[-1]))
    
        # compute the mutual information based contrast quality measure
        return mutuals(v_channel_histogram, v_channel_equalized_histogram)


    def contrast_quality(self,image):
        """
        Computes the contrast quality of image file
        :param image_url: string : path to image file resource
        :return: float : contrast quality of input image
        """
        try:
            return self.compute_contrast_quality_for_image(image)
        except IOError as err:
            return err
    

class ImageQualityScore(object):
    
    '''This class implement the BRISQUE Algorithm'''
    
    def aggd_fit(self,struct_dis):
        
        ''' AGGD fit model, takes input as the MSCN Image / Pair-wise Product'''
        
        # variables to count positive pixels / negative pixels and their squared sum
        positive_count = 0
        negative_count = 0
        positive_square_count = 0
        negative_square_count = 0
        absolute_sum=0
    
        positive_count = len(struct_dis[struct_dis > 0]) # number of positive pixels
        negative_count = len(struct_dis[struct_dis < 0]) # number of negative pixels
        
        # calculate squared sum of positive pixels and negative pixels
        positive_square_count = np.sum(np.power(struct_dis[struct_dis > 0], 2))
        negative_square_count = np.sum(np.power(struct_dis[struct_dis < 0], 2))
        
        # absolute squared sum
        absolute_sum = np.sum(struct_dis[struct_dis > 0]) + np.sum(-1 * struct_dis[struct_dis < 0])
    
        # calculate left sigma variance and right sigma variance
        left_sigma_best = np.sqrt((negative_square_count/negative_count))
        right_sigma_best = np.sqrt((positive_square_count/positive_count))
    
        gamma_hat = left_sigma_best/right_sigma_best
        
        # total number of pixels - totalcount
        total_count = struct_dis.shape[1] * struct_dis.shape[0]
    
        right_hat = maths.pow(absolute_sum/total_count, 2)/((negative_square_count + positive_square_count)/total_count)
        right_hat_norm = right_hat * (maths.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)/(maths.pow(maths.pow(gamma_hat, 2) + 1, 2))
        
        prev_gamma = 0
        prev_diff  = 1e10
        sampling  = 0.001
        gammas = 0.2
    
        # vectorized function call for best fitting parameters
        vect_function = np.vectorize(self.best_fit_param, otypes = [np.float], cache = False)
        
        # calculate best fit params
        gamma_best = vect_function(gammas, prev_gamma, prev_diff, sampling, right_hat_norm)
    
        return [left_sigma_best, right_sigma_best, gamma_best] 
    
    def best_fit_param(self, gamma, prev_gamma, prev_diff, sampling, right_hat_norm):
        
        while(gamma < 10):
            gamma_value = gammas_value(2/gamma) * gammas_value(2/gamma) / (gammas_value(1/gamma) * gammas_value(3/gamma))
            diff = abs(gamma_value - right_hat_norm)
            if(diff > prev_diff): break
            prev_diff = diff
            prev_gamma = gamma
            gamma += sampling
        gamma_best = prev_gamma
        return gamma_best
    
    def compute_features(self,img):
        
        scalenum = 2
        feat = []
        # make a copy of the image 
        im_original = img.copy()
    
        # scale the images twice 
        for itr_scale in range(scalenum):
            image = im_original.copy()
            # normalize the image
            image = image / 255.0
    
            # calculating MSCN coefficients
            muscn = cv2.GaussianBlur(image, (7, 7), 1.166)
            muscn_sq = muscn * muscn
            sigma = cv2.GaussianBlur(image*image, (7, 7), 1.166)
            sigma = (sigma - muscn_sq)**0.5
            
            # structdis is the MSCN image
            struct_dis = image - muscn
            struct_dis /= (sigma + 1.0/255)
            
            # calculate best fitted parameters from MSCN image
            best_fit_params = self.aggd_fit(struct_dis)
            # unwrap the best fit parameters 
            left_sigma_best = best_fit_params[0]
            right_sigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]
            
            # append the best fit parameters for MSCN image
            feat.append(gamma_best)
            feat.append((left_sigma_best*left_sigma_best + right_sigma_best*right_sigma_best)/2)
    
            # shifting indices for creating pair-wise products
            shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2
    
            for itr_shift in range(1, len(shifts) + 1):
                orig_arr = struct_dis
                req_shift = shifts[itr_shift-1] # shifting index
    
                # create transformation matrix for warpAffine function
                matrix = np.float32([[1, 0, req_shift[1]], [0, 1, req_shift[0]]])
                shift_arr = cv2.warpAffine(orig_arr, matrix, (struct_dis.shape[1], struct_dis.shape[0]))
                
                shifted_new_struct_dis = shift_arr
                shifted_new_struct_dis = shifted_new_struct_dis * struct_dis
                # shifted_new_structdis is the pairwise product 
                # best fit the pairwise product 
                best_fit_params = self.aggd_fit(shifted_new_struct_dis)
                left_sigma_best = best_fit_params[0]
                right_sigma_best = best_fit_params[1]
                gamma_best  = best_fit_params[2]
    
                constant = maths.pow(gammas_value(1/gamma_best), 0.5)/maths.pow(gammas_value(3/gamma_best), 0.5)
                mean_param = (right_sigma_best - left_sigma_best) * (gammas_value(2/gamma_best)/gammas_value(1/gamma_best)) * constant
    
                # append the best fit calculated parameters            
                feat.append(gamma_best) # gamma best
                feat.append(mean_param) # mean shape
                feat.append(maths.pow(left_sigma_best, 2)) # left variance square
                feat.append(maths.pow(right_sigma_best, 2)) # right variance square
            
            # resize the image on next iteration
            im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        return feat
    
    # 
    # takes input of the image path
    def test_measure_BRISQUE(self,image):
       
        '''function to calculate BRISQUE quality score '''
        
        # read image from given path
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = self.compute_features(image)
        node_array = [0]
        
        # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
        minimum = [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
        
        maximum = [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]
    
        for rescale in range(0, 36):
            min = minimum[rescale]
            max = maximum[rescale] 
            node_array.append(-1 + (2.0/(max - min) * (features[rescale] - min)))
        
        model = svmutil.svm_load_model(abs_model_path + model_path)
    
        # create svm node array from python list
        node_array, idx = gen_svm_nodearray(node_array[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
        node_array[36].index = -1 # set last index to -1 to indicate the end.
    	
    	# get important parameters from model
        svm_type = model.get_svm_type()
        is_prob_model = model.is_probability_model()
        nr_class = model.get_nr_class()
        
        if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
            nr_classifier = 1
        dec_values = (c_double * nr_classifier)()
        qualityscore = svmutil.libsvm.svm_predict_probability(model,node_array, dec_values)
    
        return qualityscore
    
   
    

ImageQualityValue=ImageQualityScore()
BlurOrNot=CheckBlur()
BrightnessAndContrastScore=ImageContrastAndBrightnessScore()
CheckResolutionOfImage=CheckResolution()
