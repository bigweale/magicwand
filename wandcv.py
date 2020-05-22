# For camera module
from picamera import PiCamera
from picamera.array import PiRGBArray

# For image processing
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

from os import system
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import time
import subprocess
import _thread
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#import pytesseract
#import keras_ocr

backSub = cv2.createBackgroundSubtractorKNN()

# initializing Picamera
camera = PiCamera()
camera.framerate = 33
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size = (640, 480))

# Define parameters for the required blob
params = cv2.SimpleBlobDetector_Params()

# creating object for SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
subprocess.Popen(['python3', 'CameraLED.py', 'off'])

mypath = '/home/pi/data/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

x = []
y = []

for file in files:
	label = file.split('_')[0]
	y.append(label)
	img = cv2.imread(join(mypath, file))
	img = cv2.resize(img, (30, 30))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret,img = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
	#cv2.imshow(file, img)
	img = img.flatten()
	x.append(img)

#print(x[0].shape)
#print(y[0])

knn = KNeighborsClassifier(n_neighbors = 5, weights='distance',metric='jaccard')
knn.fit(x,y)

#rnn = RadiusNeighborsClassifier(weights='distance', metric='matching')
#rnn.fit(x,y)

#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#clf.fit(x,y)

i=0
cpred = 'Z'
ppred = 'Z'
kernel = np.ones((5,5), np.uint8)
    
    
for image in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    frame = image.array
    frame = cv2.flip(frame, 1)
    frame = cv2.dilate(frame, kernel, iterations=2)
    frame = cv2.erode(frame, kernel, iterations=2)     
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,frame = cv2.threshold(frame, 155, 255, cv2.THRESH_BINARY)
    frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))

    fgMask = backSub.apply(frame)
    
    if i==0:
        mask = np.zeros_like(fgMask)
    else:
        mask = mask * .9
        mask = mask + fgMask
    
    i = i + 1
    #cv2.imshow("Binary Mask", mask)

    mask2 = mask.astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.dilate(mask2, kernel, iterations=2)
    mask2 = cv2.erode(mask2, kernel, iterations=2)    
    ret, mask2 = cv2.threshold(mask2, 1, 255, 0)    
    cv2.imshow("Modified Mask", mask2)

    img2,contours,hierarcy = cv2.findContours(mask2,1,1)
    if(len(contours) > 0):
        for j, cont in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cont)

            if(j==0):
                xmin = x
                ymin = y
                xmax = x+w
                ymax = y+h
            else:
                xmin = min(xmin,x)
                ymin = min(ymin,y)
                xmax = max(xmax,x+w)
                ymax = max(ymax,y+h)
                
        mask2 = mask2[ymin:ymax, xmin:xmax]
        mask2 = cv2.resize(mask2, (30, 30))
    else:
        mask2 = np.zeros((30,30), np.uint8)
            
    if i%20 == 0:
        mask2 = mask2.flatten()
        mask2 = mask2.reshape(1,-1)
        
        if(i>60):
            pred = knn.predict(mask2)
        else:
            pred = 'Z'
        print(pred)
        
        if(pred != 'Z' and pred == cpred and pred == ppred):
            if pred == 'g':
                subprocess.Popen(['tplight', 'hex', '-t', '500', '192.168.1.6', '\"ff0000\"'])
            elif pred == 'h':
                subprocess.Popen(['tplight', 'hex', '-t', '500', '192.168.1.6', '\"ffff00\"'])
            elif pred == 's':
                subprocess.Popen(['tplight', 'hex', '-t', '500', '192.168.1.6', '\"00ff00\"'])
            elif pred == 'r':
                subprocess.Popen(['tplight', 'hex', '-t', '500', '192.168.1.6', '\"0066ff\"'])
            elif pred == 'v':
                subprocess.Popen(['tplight', 'temp', '-t', '500', '192.168.1.6', '3500'])
            
            pred = 'Z'
            
        ppred = cpred
        cpred = pred  
    
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv2.destroyAllWindows()
