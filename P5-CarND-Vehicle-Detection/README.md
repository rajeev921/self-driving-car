**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[hog-car-1]: ./output_images/hog-car-1.png
[hog-car-2]: ./output_images/hog-car-2.png
[hog-noncar-1]: ./output_images/hog-noncar-1.png
[hog-noncar-2]: ./output_images/hog-noncar-2.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG (Histogram of Gradients) features were extracted using the `hog` method in `skimage.feature` package.

The code for this in the Lesson Functions section of the `Notebook.ipynb` file.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Cars**

![alt text][hog-car-1]
![alt text][hog-car-2]

**Non-Cars**

![alt text][hog-noncar-1]
![alt text][hog-noncar-2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried the HOG search with various parameter combinations and noticed that `YCrCb` color space worked very well. I use all the three channels as the feature vectors.

I use 8x8 pixels for each cell and 2x2 cells in each block. This gives a good balance of being computation heavy and extracting good feature set.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used total three feature vectors. One is spatial binning to get the raw color info, second is using a histogram of the color spectrum to get color info, and third is the HOG features to get the shape info. I concatenate all of them to become my feature vector.

The total length of the feature vector comes to 6108. It takes 3.26 seconds to train my SVC model with a test accuracy of 0.9907.

I use the `LinearSVC()` from the `sklearn.svm` package.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is an example and the corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues currently is that the size of the bounding box does not vary smoothly between frames. It needs more granularity.

Second is that when two cars are overtaking, it treats both as the same car. This is not ideal, and we'll need to track individual positions more accurately over time.

Third, the processing pipeline is still very slow, and we'd like to have it run in realtime. Currently processes 3 frames per second.
