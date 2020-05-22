# Magic Wand

A simple project to control a IOT lightbulb by tracking a reflected infrared point. This project includes the base python file and some sample data files for training.

[![Image of Wand](https://i9.ytimg.com/vi/AUtWAyYNayg/mq2.jpg?sqp=CObqn_YF&rs=AOn4CLC_sWlJ0ITZxeWbY_100ZyrVWb3wQ)](https://www.youtube.com/embed/AUtWAyYNayg)

## Hardware
* Raspberry Pi 3 Model B Plus (Rev 1.3)
* [AdruCAM NOIR 5MP OV5647](https://smile.amazon.com/dp/B083514WC2/ref=cm_sw_em_r_mt_dp_U_CT-XEb9GKXEKH)
* [Point-Based IR Reflective Device](https://shop.universalorlando.com/c/Harry-Potter-Interactive-Wands.html)

## Software
* [Raspbian](https://www.raspberrypi.org/downloads/raspbian/)
* Python 3 with the following modules installed:
  * [PiCamera](https://picamera.readthedocs.io/en/release-1.13/)
  * [OpenCV](https://opencv.org/)
  * [numpy](https://numpy.org/)
  * [Scikit-learn](https://scikit-learn.org/stable/index.html)
* [TP-Link Software](https://www.npmjs.com/package/tplink-lightbulb)
* [ArduCAM Software](https://github.com/ArduCAM/RPI_Motorized_IRCut_Control): Helps provide a more-stable camera stream by disabling IR cutover. Place _CameraLED.py_ in the same directory as the _wandcv.py_ file.

## Computer Vision Algorithms
Camera input is converted to grayscale and thresholded in order to extract only the more-prevelant features from the IR camera. We use a simple [KNN background subtraction](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html) in order to remove stable background noise. Dialation and eroding are used to help connect neighboring points.

We use a simple accumulator-type object to create basic 'motion masks' in order to track the historical path of the IR points. Historical value decay has been set at _.9_ based on trial-and-error.

Once we have a history, we create a bounding box that surrounds all detected motion in the image. The portion of the input denoted by the bounding box is extracted and then scaled to 30x30 for inference (described below).

The bounding box is succeptable any noise that remains after thresholding and background substraction. More robust image processing capabilites would improve the stability of the extracted movement patterns.

## Machine Learning Algorithms
We use a simple [K-Nearest Neighbors algorithm](https://scikit-learn.org/stable/modules/neighbors.html) to predict the input, given a set of training files.

### Training:
Included in the repository is a set of training images. There are six training examples across five different types of classes and three examples for the base class. Each file is 100x100. The files are named <class>_#.jpg and should be successfully parsed and trained by the python file.

Classes:
* __s:__ Turns the light green; looks for 's'
* __h:__ Turns the light yellow; looks for 'h'
* __g:__ Turns the light red; looks for 'G', also works with 'O'
* __r:__ Turns the light blue; looks for an inverted 'V'; the 'R' pattern was too close to the 'h' pattern
* __v:__ Resets the light to a normal color
* __Z:__ null (completely blank input). Makes no changes.

### Inference
The normalized and scaled foreground motion mask is compared to the training files by using five (5) nearest neighbors and the [jaccard distance metric](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric). These were chosen using trial-and-error methodology. More data and a more-rigorous approach could improved inference stability.
