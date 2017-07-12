import numpy as np
import cv2
import math
import pickle
import os
import matplotlib.pyplot as plt

mtx = None
dist = None

#########################################################
#														#
#Load pickled camera matrix and distortion coefficients #
#														#
#########################################################

def camera_calibration_params():
    with open('camera_dist_pickle.p', mode='rb') as f:
        dist_pickle = pickle.load(f)
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    return mtx, dist

#########################################################
#														#
#               Undistorted Image                       #
#														#
#########################################################

def undistort_img(test_img):
    mtx, dist = camera_calibration_params()
    img = cv2.undistort(test_img, mtx, dist, None, mtx)
    return img


#########################################################
#														#
#               Binarize image                         #
#														#
#########################################################
	
def sobel_binary_img(img, orient='x', thresh = (20, 100)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary = np.zeros_like(scaled_sobel)  
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary
	
def color_thresh_img(img, orient='x', s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the l and s channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
        
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= s_thresh[0]) & (l_channel <= s_thresh[1])] = 1

    return s_binary	
	
def binary_image(img, s_thresh= (90, 255), l_thresh= (40, 255), ksize_sx=3 , sx_thresh=(30, 100)):
    
    # Convert to HSV color space and separate the V channel   
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    
    # Convert to HLS color space and separate the l and s channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Normaliza the l_channel and v_channel
    l_channel = (l_channel - np.min(l_channel)) * 255. / (np.max(l_channel) - np.min(l_channel))
    v_channel = (v_channel - np.min(v_channel)) * 255. / (np.max(v_channel) - np.min(v_channel))
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=ksize_sx) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    sxbinary[(v_channel < 137)] = 0
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    binary = np.zeros_like(l_binary)
    binary[(l_binary == 1) & (s_binary == 1) | (sxbinary == 1) | (v_channel > 220)] = 1
    
    color_binary = np.dstack((l_binary, sxbinary, s_binary))
    binary = (np.dstack(( binary, binary, binary))*255.).astype('uint8')
    
    return binary
    	

#########################################################
#														#
#            Perspective Transform                      #
#														#
#########################################################
def drawQuad(image, points, color=[255, 0, 0], thickness=4):
	p1, p2, p3, p4 = points
	cv2.line(image, tuple(p1), tuple(p2), color, thickness)
	cv2.line(image, tuple(p2), tuple(p3), color, thickness)
	cv2.line(image, tuple(p3), tuple(p4), color, thickness)
	cv2.line(image, tuple(p4), tuple(p1), color, thickness)


def warp_image(image, debug=False, size_top=70, size_bottom=370):
	height, width = image.shape[0:2]
	output_size = height/2

	src = np.float32([
					[(width/2) - size_top, height*0.65],
					[(width/2) + size_top, height*0.65], 
					[(width/2) + size_bottom, height-50],
					[(width/2) - size_bottom, height-50]
					])
	dst = np.float32([
					[(width/2) - output_size, (height/2) - output_size], 
					[(width/2) + output_size, (height/2) - output_size], 
					[(width/2) + output_size, (height/2) + output_size], 
					[(width/2) - output_size, (height/2) + output_size]
					])
    
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
	
	if debug:
		drawQuad(image, src, [255, 0, 0])
		drawQuad(image, dst, [255, 255, 0])
		plt.imshow(image)
		plt.show()
		
	return warped
#########################################################
#														#
# Initialization: loads camera calibration parameters.  #
#														#
#########################################################

def main():
	camera_calibration_params()
    