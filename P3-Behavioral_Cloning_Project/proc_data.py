from os.path import join
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config as cf



#
# Loads data as numpy array from driving_log.csv file.
# Params: csv_driving_data - path to the driving_log.csv file.
# Returns: numpy array of loaded rows from csv file.
#
def load_data(csv_driving_data, min_throttle=0.0, verbose=False):
    data_frame = pd.DataFrame.from_csv(csv_driving_data, index_col=None)
    data = data_frame.values
    header = data_frame.columns.values
    assert(header[cf.CENTER] == 'center')
    assert(header[cf.LEFT] == 'left')
    assert(header[cf.RIGHT] == 'right')
    assert(header[cf.STEERING] == 'steering')
    assert(header[cf.THROTTLE] == 'throttle')
    
    if verbose:
        print('data shape: ', data.shape)
        print('driving_log header: ', header)
        print(data[0:2])
    return data
    

#
# Splitting of data set to training and validation sets.
# Params: data - data - array of records from driving_log.csv file.
# Returns: training and validation arrays.
#
def train_validation_split(data, test_size=0.2):
    """
    Splits data, loaded from csv driving_log file, into training and validation sets
    param data: numpy array (or list) from Udacity csv driving_log
    return: train & validation sets:
    """
    data_train, data_val = train_test_split(data, test_size=test_size, random_state=0)
    return data_train, data_val

    
#
# Loads image from data folder for specified sample and camera center/left/right.
# Returned steering angle is corrected for images from left and right cameras to
# simulate scenarios when the car drives to the center of road from the left
# and right sides of road.
# Params: data - data - array of records from driving_log.csv file.
#         idx - index of sample in data (index row in driving_log.csv file).
#         cam_idx - camera index to specify from which camera takes the image.
#         data_dir - path to the driving_log.csv file.
# Returns: requested RGB image, steering angle, is corrected for images from
#          left and right cameras. 
#
def read_sample(data, idx, cam_idx, data_dir=cf.DATA_FOLDER):
    sample = data[idx]

    f_path = None

    steering = sample[cf.STEERING]
    #throttle = sample[cf.THROTTLE]
    
    if (cam_idx == cf.CENTER):
        f_path = sample[cf.CENTER].strip()
    elif (cam_idx == cf.LEFT):
        f_path = sample[cf.LEFT].strip()
        steering = steering + cf.CAM_STEERING_SHIFT
    elif (cam_idx == cf.RIGHT):
        f_path = sample[cf.RIGHT].strip()
        steering = steering - cf.CAM_STEERING_SHIFT
    else:
        print('Invalid camera index:',  cam_idx)
        
    img = cv2.imread(join(data_dir, f_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, steering

    
#
# Cropping of RGB image (to remove part of sky and bonnet), resizing to the
# size of input layer of the model. 
# Params: img - source RGB image.
# Returns: cropped and resized Gray/RGB image.
#
def crop_image(img, verbose=False):    
    assert(img.shape[1] == 320)
    assert(img.shape[0] == 160)

    imgR = None
    
    if cf.IN_Channels == 1:    
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgR = cv2.resize(gray, (cf.IN_Width, cf.IN_Height), interpolation=cv2.INTER_AREA)
    
        if verbose:
            plt.figure(1), plt.imshow(img)
            plt.figure(2), plt.imshow(gray)
            plt.figure(3), plt.imshow(imgR)
            plt.show()        
    
        imgR = np.expand_dims(imgR, axis=2)
    else:        
        imgC = img[cf.IN_Crop, :, :]
        imgR = cv2.resize(imgC, (cf.IN_Width, cf.IN_Height), interpolation=cv2.INTER_AREA)
        
        if verbose:
            plt.figure(1), plt.imshow(img)
            plt.figure(2), plt.imshow(imgC)
            plt.figure(3), plt.imshow(imgR)
            plt.show()        
        
    return imgR
    


#
# Random displacement in horizontal and vertical dirrections.
# Horizontal displacement allows to learn scenarios when car drives along left or right sides of road.
# Vertical displacement is used to get adaptation to horizont variation.
# Params: img - source RGB image.
#         steering - source steering angle.
#         horz_range - range of horiz. displacement.
#         vert_range - range of vertic. displacement.
# Returns: shifted RGB image, corrected steering angle for horizontal shift.
#
def random_translate(img, steering, horz_range=30, vert_range=5):
    rows, cols, chs = img.shape
    tx = np.random.randint(-horz_range, horz_range+1)
    ty = np.random.randint(-vert_range, vert_range+1)
    #print('translate: ', tx, ty)
    steering = steering + tx * 0.004 # mul by steering angle units per pixel
    tr_M = np.float32([[1,0,tx], [0,1,ty]])
    img = cv2.warpAffine(img, tr_M, (cols,rows), borderMode=1)
    return img, steering 


#
# Adding random variation of brightness to get adaptation to different light conditions.
# Params:  img - source RGB image.
# Returns: RGB image with modified brightness.
#   
def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype('float32')
    #print(hsv.dtype)
    rand_bright = .25+np.random.uniform()
    #print('rand_bright=', rand_bright)
    hsv[:,:,2] = hsv[:,:,2] * rand_bright;
    hsv[:,:,2] = np.clip(hsv[:,:,2], a_min=0, a_max=255)
    hsv = hsv.astype('uint8')
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

 

#
# Random horizontal flipping of RGB image.
# Params: img - source RGB image.
#         steering: source steering angle.
# Returns: flipped RGB image and inverted steering angle.
#
def random_horz_flip(img, steering):
    i_flip = np.random.randint(2)
    if i_flip==0:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering    
    
    

# not used now
def generate_train_sample(data, s_i, verbose=False):
    if verbose:
        print('train sample s_i:', s_i)
        
    cam_i = np.random.randint(3)
    if verbose:
        print('camera index:', cam_i)
    
    img, steering = read_sample(data, s_i, cam_i)
    
    if verbose:
        print('steering:', steering)
        plt.imshow(img)
        
    img, steering = random_translate(img, steering)
        
    img, steering = random_horz_flip(img, steering)

    return img, steering


    
# Generates batch of specified size for training or validation.
# Params: data - array of records from driving_log.csv file.
#         batch_size - size of generated batch.
#         bias - value [0..1] to control frequency of samples with steering angle close
#                to zero in batch.
#         augment - enable/disable augmentation.
def generate_batch(data, batch_size=32, bias=1.0, augment=True):
    #
    # Augmentation: using all three cameras to learn the following scenarious:
    #       driving in the center of road, driving from left/right part of road
    #       back to the center, driving along left/right part of road (commented now);
    #       random horizontal fliping; variation of brightness to get adaptation
    #       to different light conditions.
    #
    # Bootstrapping approach is used to generate batch of any size with the same
    # measures of accuracy (in terms of bais, varience, etc.)- random sampling
    # with replacement.
    #
    batch_im = np.zeros((batch_size, cf.IN_Height, cf.IN_Width, cf.IN_Channels))
    batch_st = np.zeros(batch_size)
    
    n_samples = 0
    
    while n_samples < batch_size:
        idx = np.random.randint(len(data))
        
        cam_i = cf.CENTER
        if augment:
            # Random selection of camera for sample
            cam_i = np.random.randint(3)
            
        #print('camera index:', cam_i)
        img, steering = read_sample(data, idx, cam_i)
        #print('steering: ', steering)
        
        if augment:

            img = random_brightness(img)
            img, steering = random_horz_flip(img, steering)
            img, steering = random_translate(img, steering)
                    
        #
        # Bias parameter [0..1] allows to control the probability of
        # selecting samples with steering angle close to zero for training,
        # to exclude bias of prediction toward zero.
        #
        # bias=1 to passes all samples with steering angle close to 0.
        # bias=0 to drop all steering values close to 0.
        #
        steering_thresh = np.random.rand()
        if (abs(steering) + bias) < steering_thresh:
            pass # drop this sample
        else:
            img = crop_image(img)
            batch_im[n_samples] = img
            batch_st[n_samples] = steering
            n_samples += 1

    return batch_im, batch_st



#
# Preprocessing of input frame for prediction: cropping and resizing of
#  image to the size of input layer of the model.
#
def preprocess_predict_image(img, verbose=False):
    return crop_image(img, verbose)
    

#
# Returns generator of batch for training of the model.
# Params: data - array of records from driving_log.csv file.
#         batch_size - size of generated batch.
#            
def generate_train_data_batch(data, batch_size=32, bias=0.8, augment=True, pb_thresh=0.1):
    while 1:
        batch_imgs, batch_steering = generate_batch(data, batch_size, bias, augment)
        yield  batch_imgs, batch_steering    
        
        
#       
# Returns generator of batch for validation of the model.
# Params: data - array of records from driving_log.csv file.
#
def generate_valid_data(data):
    while True:
        for i in range(len(data)):
            x,y = read_sample(data, i, cf.CENTER)
            x = crop_image(x)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = np.array([[y]])
            yield x, y
