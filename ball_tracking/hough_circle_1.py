import cv2
import numpy as np
import argparse
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

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	gframe = cv2.medianBlur(frame,5)
	gframe = cv2.cvtColor(gframe,cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gframe,cv2.HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=5,maxRadius=50)

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
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()