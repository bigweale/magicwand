# Basic Python Imports
import argparse, sys
import time
import subprocess

# Data Set File Handling
from os import system
from os import listdir
from os.path import isfile, join

# Camera Module
from picamera import PiCamera
from picamera.array import PiRGBArray

# Computer Vision and Machine Learning Modulese
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Parse IP address from command line
parser = argparse.ArgumentParser(description='Control a light with an IR camera.')
parser.add_argument('ip address', metavar='IP', type=str, help='IP address of the TPLink bulb')
args = parser.parse_args()
light_ip = sys.argv[1]

# Initialize PiCamera
camera = PiCamera()
camera.framerate = 30
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size = (640, 480))

# Ensure that the camera is in 'night vision' mode
subprocess.Popen(['python3', 'CameraLED.py', 'off'])

# Initialize Vision Background Model and Kernel for Expansion/Dilation
backSub = cv2.createBackgroundSubtractorKNN()
kernel = np.ones((5,5), np.uint8)

## READ TRAINING DATA FOR KNN Detection ##

x = []
y = []

# Find all files in the data directory
mypath = '/home/pi/data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Files are in the format <label>_<int>.jpg
for file in files:
    
    # Split label from filename and add to training set
    label = file.split('_')[0]
    y.append(label)
    
    # Read, resize and normalize images
    img = cv2.imread(join(mypath, file))
    img = cv2.resize(img, (30, 30))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    x.append(img)

# Train KNN using the input images
knn = KNeighborsClassifier(n_neighbors = 5, weights='distance',metric='jaccard')
knn.fit(x,y)

# Initialize Variables
inc=0
curr_pred = 'Z'
prev_pred = 'Z'

## MAIN IMAGE PROCESSING LOOP ##
for image in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    # Convert raw camera capture to a thresholded, binary image
    frame = image.array
    frame = cv2.flip(frame, 1)
    frame = cv2.dilate(frame, kernel, iterations=2)
    frame = cv2.erode(frame, kernel, iterations=2)     
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,frame = cv2.threshold(frame, 155, 255, cv2.THRESH_BINARY)
    frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
    
    # Apply background subtraction to get the foreground image
    foreground = backSub.apply(frame)
    
    ## Use a decaying 'movement mask' to track history of the foreground
    if inc==0:
        #Initialize on the first frame
        mMask = np.zeros_like(foreground)
    else:
        # Decay previous movement mask and then add the new foreground
        mMask = mMask * .9
        mMask = mMask + foreground

    # Increment the frame counter
    inc = inc + 1

    # Dilate and Erode the movement mask
    mMask2 = mMask.astype(np.uint8)
    mMask2 = cv2.dilate(mMask2, kernel, iterations=2)
    mMask2 = cv2.erode(mMask2, kernel, iterations=2)    
    ret, mMask2 = cv2.threshold(mMask2, 1, 255, 0)    

    # Show movement mask for testing
    #cv2.imshow("Modified Mask", mask2)

    # Extract the bounding box containing all movement points in the mask
    img2,contours,hierarcy = cv2.findContours(mMask2,1,1)
    if(len(contours) > 0):
        # Movement found, find bounding box
        for j, cont in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cont)

            if(j==0):
                # Initialize box to first rectangle
                xmin = x
                ymin = y
                xmax = x+w
                ymax = y+h
            else:
                # Check for expanded rectangle dimensions
                xmin = min(xmin,x)
                ymin = min(ymin,y)
                xmax = max(xmax,x+w)
                ymax = max(ymax,y+h)
        
        # Convert full image to bounded image and rescale
        mMask2 = mMask2[ymin:ymax, xmin:xmax]
        mMask2 = cv2.resize(mMask2, (30, 30))
    else:
        # No movement found, output zeros
        mMask2 = np.zeros((30,30), np.uint8)

    ## Test for a valid pattern three times every two seconds
    if inc % 20 == 0:
        pred = 'Z'
        mMask2 = mMask2.flatten()
        mMask2 = mMask2.reshape(1,-1)
        
        # Predict only if camera has been on for two seconds to stablilize background model
        if(inc > 60):
            pred = knn.predict(mMask2)

        # For debugging
        #print(pred)
        
        # Perform an action ONLY if the prediction has been consistent across 2 seconds
        if(pred != 'Z' and pred == curr_pred and pred == prev_pred):
            if pred == 'g':
                subprocess.Popen(['tplight', 'hex', '-t', '500', light_ip, '\"ff0000\"'])
            elif pred == 'h':
                subprocess.Popen(['tplight', 'hex', '-t', '500', light_ip, '\"ffff00\"'])
            elif pred == 's':
                subprocess.Popen(['tplight', 'hex', '-t', '500', light_ip, '\"00ff00\"'])
            elif pred == 'r':
                subprocess.Popen(['tplight', 'hex', '-t', '500', light_ip, '\"0066ff\"'])
            elif pred == 'v':
                subprocess.Popen(['tplight', 'temp', '-t', '500', light_ip, '3500'])
            
            pred = 'Z'
            
        prev_pred = curr_pred
        curr_pred = pred
    
    rawCapture.truncate(0)

    # end if the user clicks 'q' on the display (only with imshow enabled)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
#fin
