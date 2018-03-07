import cv2
import numpy as np
import argparse
import imutils
import sys

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# define the lower and upper boundaries of the "orange"
# ball in the HSV color space, then initialize the
# list of tracked points

h1 = 35; h2 = 85
s1 = 70; s2 = 255
v1 = 30; v2 = 255


yellowLower = (h1, s1, v1)
yellowUpper = (h2, s2, v2)

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=1000)
	blur_frame = cv2.medianBlur(frame, 25)
	hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "yellow", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask1 = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask2 = cv2.erode(mask1, None, iterations=2)
	mask3 = cv2.dilate(mask2, None, iterations=3)
	# mask4 = cv2.dilate(mask3, None, iterations=2)
	mask5 = cv2.erode(mask3, None, iterations=3)

	# bitwise and to take hsv mask and place it over the
	# original
	res = cv2.bitwise_and(frame, frame, mask=mask5)

	gray_frame = cv2.cvtColor(blur_frame,cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray_frame,cv2.HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=5,maxRadius=50)

	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			cv2.circle(frame,(x, y),r,(0,255,0),1) # draw the outer circle
			cv2.circle(frame,(x, y),2,(0,0,255),3) # draw the center of the circle


	# show the frame to our screen
	cv2.imshow('detected circles', frame)
	cv2.imshow('Hough Circles frame', res)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()