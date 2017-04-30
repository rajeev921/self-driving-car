import os
import glob
import pickle
import numpy as np
import cv2

# Load the images and save the images
images = glob.glob('./camera_cal/calibration*.jpg')
out_dir = './output_images/'

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

mtx = None
dist = None
	
#########################################################
#														#
#                  Caliberate Images                    #
#														#
#########################################################

def calibration():

	# Make a list of calibration images
	nx = 9 # number of inner conners along X
	ny = 6 # number of inner conners along Y

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

	# Step through the list and search for chessboard corners
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
   
		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
			image_name = os.path.split(fname)[1]
			write_name = out_dir+ image_name
			#print(write_name)
			cv2.imwrite(write_name, img)
			#cv2.imshow('img',img)
			cv2.waitKey(500)
    
	#cv2.destroyAllWindows()

	
#########################################################
#														#
#              Undistorted Images                       #
#														#
#########################################################
	
def test_undistort():
	# Test undistortion on an image
	img = cv2.imread('camera_cal/calibration1.jpg')
	img_size = (img.shape[1], img.shape[0])

	# Do camera calibration given object points and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)

	
#########################################################
#														#
#Save pickled camera matrix and distortion coefficients #
#														#
#########################################################
	
def save_matrix():
	# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
	dist_pickle = {}
	dist_pickle["mtx"] = mtx   # camera Matrix
	dist_pickle["dist"] = dist # Distortion Coefficient
	pickle.dump( dist_pickle, open( "camera_dist_pickle.p", "wb" ) )
	print("Done")	
	
def init():
	
	calibration()
	test_undistort()
	save_matrix()

if __name__ == "__main__":
	init()
	