# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "yellow"
# ball in the HSV color space, then initialize the
# list of tracked points

h1 = 35; h2 = 85
s1 = 70; s2 = 255
v1 = 30; v2 = 255


yellowLower = (h1, s1, v1)
yellowUpper = (h2, s2, v2)

pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
# first_run = True
while True:


	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=1000)
	blur_frame = cv2.medianBlur(frame, 5)
	hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "yellow", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask1 = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask2 = cv2.erode(mask1, None, iterations=2)
	mask3 = cv2.dilate(mask2, None, iterations=3)
	# mask4 = cv2.dilate(mask3, None, iterations=2)
	mask5 = cv2.erode(mask3, None, iterations=3)


	# if first_run:
	# 	mask_cycle = 1
	# 	mask = mask5
	# 	first_run = False

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask3, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)

		############################################
		# Intermittent divide by 0 error!!
		############################################
		if  M["m00"] != 0:
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 1:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# Bitwise-AND mask and original image
	cv2.imshow('frame', frame)
	cv2.imshow('hsv', hsv)
	cv2.imshow('mask', mask2)
	key = cv2.waitKey(1) & 0xFF

	# show the frame to our screen

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
	elif key == ord('2'):
		h1 += 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('w'):
		h1 -= 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('s'):
		h2 += 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	elif key == ord('x'):
		h2 -= 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	elif key == ord('3'):
		s1 += 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('e'):
		s1 -= 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('d'):
		s2 += 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	elif key == ord('c'):
		s2 -= 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	elif key == ord('4'):
		v1 += 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('r'):
		v1 -= 1
		yellowLower = (h1, s1, v1)
		print("yellowLower = " + str(yellowLower))
	elif key == ord('f'):
		v2 += 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	elif key == ord('v'):
		v2 -= 1
		yellowUpper = (h2, s2, v2)
		print("yellowUpper = " + str(yellowUpper))
	# elif key ==ord('`'):
	# 	if mask_cycle == 1:
	# 		mask = mask2
	# 		mask_cycle = 2
	# 		cv2.imshow('mask', mask)
	# 		print("mask_cycle = " + str(mask_cycle))
	# 	elif mask_cycle == 2:
	# 		mask = mask3
	# 		mask_cycle = 3
	# 		cv2.imshow('mask', mask)
	# 		print("mask_cycle = " + str(mask_cycle))
	# 	elif mask_cycle  == 3:
	# 		mask = mask5
	# 		mask_cycle = 5
	# 		print("mask_cycle = " + str(mask_cycle))
	# 	elif mask_cycle == 5:
	# 		mask = mask1
	# 		mask_cycle = 1
	# 		print("mask_cycle = " + str(mask_cycle))

		# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
