
# Project 3 - Behavioral Cloning with Keras

The main goal of this project is developement and training a Deep Network to clone human behavioral during driving a car, thus being able to drive autonomously on a simulator, provided by Udacity. The deep network drives the car autonomously along a track, using the image from cenral camera to predict steering angle at each instant.

## General description.

The simulated car is equipped with three cameras, one on the left side, one in the center and one on the right side of the car, that provide images from these different view points. Simulator supports two modes: training mode and autonomous mode. Training mode allows to use all three cameras for recording dataset, used for model training. In autonomous mode only central camera provides images for steering angle prediction. The training Track 1 has bridge, sharp corners, exits, entries, partially missing lane lines of the road bends. An additional test Track 2 for testing the model, has changing elevations, even sharper turns and bumps. It is thus crucial that the Deep Network not only memorize the first track during training, but able to generalize the unseen data to perform correct prediction of steering angles to drive the Track 2 successfully. The developed model was trained only on the training Track 1 and it successfully drives on the test Track 2. The model was trained on Udacity dataset, which contains recorded one lap of the Track 1.

## Dataset

The model was trained on Udacity dataset, containing 8036 samples. Each sample provides the following data, as you can see from the table below:
<ol>
<li>images from three cameras</li>
<li>corresponding steering angle</li>
<li>throttle</li>
<li>break</li>
<li>speed value</li>
</ol>
![data_records](imgs/driving-log-output.png)

#### Image samples

Below you can see typical images from three cameras in each sample record.

![camera_images](imgs/camera_samples.png)

#### Data distribution

Udacity dataset contains record of one lap with human driving along the middle of the road. The analysis of the dataset shows, that the dataset has huge skew in the ground truth data distribution: the steering angle distribution is strongly biased towards the straight direction. 

![training_data_before_preprocessing](imgs/histogram_udacity_dataset.png)

Without accounting for this bias towards the straight direction, the model will not ever learn what to do if it gets off to the side of the road, and as result, the car usually leaves the tack quickly. One way to counteract this problem is to add record data to the dataset when the car is driving from the side of the road back toward the center line.

## Preparation of training dataset

The model was trained on Udacity dataset, containing 8036 samples. As it mentioned above, the first problem with the source dataset is huge bias towards the zero direction. The second problem is small size of training data for training Deep Network with large capacity. To solve these two problems and to improve generalization property of model to drive on unseen tracks, bootstrapping approach (random sampling with replacement) with data augmentation is used to generate a batch with requested size during the training of the model.  

### Data augmentation
<ol>
<li>Exploting all three cameras </li>
<li>Variation of brightness</li>
<li>Horizontal flipping</li>
<li>Horizontal and vertical shifting</li>
<li>Shifting the bias</li>
<li>Image cropping</li>
</ol>

#### Exploting all three cameras
Randomly, the image from one of the center, left or right cameras is used with correction of steering direction as a training sample. This approach, reported in [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf), allows the model to learn scenarios recovering during driving from the left or right sides of road to the middle of the road.

#### Variation of brightness
Variation of brightness increases variation in trained data to get the model more robust to different light conditions.

#### Horizontal flipping
Horizontal flipping increases variation in trained data and allows to learn the model scenarious to drive along left or right sides of the road with further smooth recovering to the middle of the road. Small vertical flipping is used to increase variation in trained data too with some robustness to horizontal variations.

#### Shifting the bias
The bias parameter with values in range [0, 1] and with random uniform thresholding was added for steering angles to tune the probability of dropout samples with steering angle close to zero from generated trained batch. The effect of bias parameter is demonstrated below on histogram of generated train batch from 2048 samples.
![training_data_after_preprocessing](imgs/histogram_data_steering_angles.png)

#### Image cropping
The preprocessed image is cropped to the size 160x80 of input layer used in the model. The bonnet of the car and half part of sky is removed during cropping too.

## Model architecture

To predict steering angles, a CNN with 730,033 parameters was developed: 3 sequentially connected convolutional layers, 3 full connected layers and one fully connected neuron. The CNN architecture is presented in the table below. The input RGB image is resized to the size of input layer: 160x80, RGB channels. The image normalization to the range [-0.5, 0.5] is implemented in the model as a lambda layer. The output neuron regresses the correct steering value from the features it receives from the previous layers. All three convolutional layers use 3x3 filter with stride equal to 1. The choice of using ELU activation function, instead of more traditional ReLU, come from model of [CommaAI](https://github.com/commaai/research/blob/master/train_steering_model.py), which was developed for the same task of steering regression.

To prevent overfitting of model, three dropout layes were added with drop probability 0.5.


|     Layer (type)     |    Output Shape     | Param #                        
|----------------------|---------------------|-----------
|  Lambda_1 (Lambda)   | (None, 80, 160, 3)  |   0                     
|  Convolution2D       | (None, 78, 158, 16) |   448         
|  ELU (Activation)    | (None, 78, 158, 16) |   0           
|  Maxpooling (2x2)    | (None, 39, 79, 16)  |   0           
|  Convolution2D       | (None, 37, 77, 32)  |   4640        
|  ELU (Activation)    | (None, 37, 77, 32)  |   0           
|  Maxpooling (2x2)    | (None, 12, 25, 32)  |   0           
|  Convolution2D       | (None, 10, 23, 48)  |   13872       
|  ELU (Activation)    | (None, 10, 23, 48)  |   0           
|  Maxpooling (2x2)    | (None, 5, 11, 48)   |   0           
|  Flatten             | (None, 2640)        |   0           
|  Dropout             | (None, 2640)        |   0           
|  Dense               | (None, 256)         |   676096      
|  ELU (Activation)    | (None, 256)         |   0           
|  Dropout             | (None, 256)         |   0           
|  Dense               | (None, 128)         |   32896       
|  ELU (Activation)    | (None, 128)         |   0           
|  Dense               | (None, 16)          |   2064        
|  ELU (Activation)    | (None, 16)          |   0           
|  Dense               | (None, 1)           |   17                             
||||
|Total params: 730,033


## Training the model

The model was compiled with Adam optimizer with default parameters and specified learning rate, equal to 0.001. The model was trained on desktop computer with 4 cores (8 logical cores) on Windows 10.

The bias parameter was fixed as 0.8 but this parameter can be changed along epochs to improve learning performance. The initial dataset is splitted on training and validation sets as 80% vs 20%. The model was trained on 10 epochs, every epoch has 20224 generated samples. Every batch with 256 samples is generated from training dataset with data  augmentation, as described above. 


## Validation the model

The model was evaluated on validation set, 20% of Udacity dataset. Only central camera without any data augmentation is used during generation of validation set.

## Testing the model and Results

After the training the model, when the car smoothly drives along training Track 1, the network can successfully drives  along unseen testing Track 2 too.

A video of the test track performance is shown below.

[![ScreenShot](imgs/track1.jpg)](https://youtu.be/YnP_kDSxEf8)

The performance of the same CNN on the training track is shown below.

[![ScreenShot](imgs/track2.jpg)](https://youtu.be/HE_y7rX2Izo)


## Conclusions

Data augmentation, with according steering angle updates to generate samples for different learning scenarious, allows to train a neural network to recover the car from extreme events, like different row conditions, different lighting conditions, by just simulating such events from regular driving data.

The developed deep network can be improved and redesigned to the aim of improvement generalization of model to drive in more difficult conditions. One of the way to experiment with construction of model, is to use RNN architecture with elements of VGG16 for features extraction to predict next element in context of previous element.
