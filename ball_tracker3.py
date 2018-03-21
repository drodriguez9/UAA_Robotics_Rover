import cv2
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import math
import numpy as np
import sys
import imutils

TENNIS_BALL_DIAMETER = 68.6 #millimeters
ZED_FOCAL_LENGTH = 2.8 #millimeters

def main():
	# Create a PyZEDCamera object
	zed = zcam.PyZEDCamera()

	# Create a PyInitParameters object and set configuration parameters
	init_params = zcam.PyInitParameters()
	init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
	init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)

	# Open the camera
	if not zed.is_opened():
		print("Opening ZED Camera...")
	status = zed.open(init_params)
	if status != tp.PyERROR_CODE.PySUCCESS:
		print(repr(status))
		exit()

	# Create and set PyRuntimeParameters after opening the camera
	runtime_parameters = zcam.PyRuntimeParameters()
	runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

	# Update user with information
	print_camera_information(camera=zed)

	# define the lower and upper boundaries of the "yellow"
	# ball in the HSV color space, then initialize the
	# list of tracked points

	h1 = 35; h2 = 85
	s1 = 70; s2 = 255
	v1 = 30; v2 = 255
	yellow_lower = (h1, s1, v1)
	yellow_upper = (h2, s2, v2)

	# Capture images and depth until 'q' is pressed
	left_image = core.PyMat()
	depth = core.PyMat()
	point_cloud = core.PyMat()

	key = ''
	while key != 113:  # for 'q' key
		# grab the current frame
		if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
			# Retrieve left image
			zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
			cv_frame = left_image.get_data()

			# Retrieve depth map. Depth is aligned on the left image
			zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)

			# Retrieve colored point cloud. Point cloud is aligned on the left image.
			zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)

			# resize the frame, blur it, and convert it to the HSV
			# color space
			# cv_frame = imutils.resize(cv_frame, width=500)
			blur_frame = cv2.medianBlur(cv_frame, 1)
			hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
	
			# construct a mask for the color "yellow", then perform
			# a series of dilations and erosions to remove any small
			# blobs left in the mask
			mask1 = cv2.inRange(hsv, yellow_lower, yellow_upper)
			mask2 = cv2.erode(mask1, None, iterations=2)
			mask3 = cv2.dilate(mask2, None, iterations=3)
			mask4 = cv2.erode(mask3, None, iterations=3)
	
			# bitwise-AND to take hsv mask and place it over the original
			res = cv2.bitwise_and(cv_frame, cv_frame, mask=mask4)

			# find contours in the mask
			cnts = cv2.findContours(mask4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

			# only proceed if at least one contour was found
			if len(cnts) > 0:
				# only look at the largest contour
				c = max(cnts, key=cv2.contourArea)
				# find the smallest circle around the largest contour
				((x, y), radius) = cv2.minEnclosingCircle(c)
				m = cv2.moments(c)
				distance_to_ball = distance_to_largest_contour(moment=m, point_cloud_matrix=point_cloud)

				if distance_to_ball != -1:
					tennis_ball_pixels_x, tennis_ball_pixels_y = size_tennis_ball_pixels(zed=zed, distance=distance_to_ball)

					# Can't have non-integer pixels, so take the ceiling of the pixels/2 to create a Region of Interest (ROI)
					x1 = math.ceil(x - tennis_ball_pixels_x / 2)
					x2 = math.ceil(x + tennis_ball_pixels_x / 2)
					y1 = math.ceil(y - tennis_ball_pixels_y / 2)
					y2 = math.ceil(y + tennis_ball_pixels_y / 2)
					width = left_image.get_width()
					height = left_image.get_height()

					if y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0:
						if x1 != x2 or y1 != y2:
								tennis_ball = cv_frame[y1:y2, x1:x2]
								cv2.imshow('test', cv_frame[y1:y2, x1:x2])
					#cv2.imshow('tennis ball', tennis_ball)

					# # Deal with ceiling math error by checking for odd number of pixels
					# if tennis_ball_pixels_x % 2 == 0 :
					# 	if tennis_ball_pixels_y % 2 == 0 :
					# 		cv_frame[0 : tennis_ball_pixels_x, 0 : tennis_ball_pixels_y] = tennis_ball
					# 	else :
					# 		cv_frame[0: tennis_ball_pixels_x, 0: tennis_ball_pixels_y + 1] = tennis_ball
					# else :
					# 	if tennis_ball_pixels_y % 2 == 0 :
					# 		cv_frame[0 : tennis_ball_pixels_x + 1, 0 : tennis_ball_pixels_y] = tennis_ball
					# 	else :
					# 		cv_frame[0: tennis_ball_pixels_x + 1, 0: tennis_ball_pixels_y + 1] = tennis_ball

			# gray_frame = cv2.cvtColor(blur_frame,cv2.COLOR_BGR2GRAY)
			# circles = cv2.HoughCircles(gray_frame,cv2.HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=5,maxRadius=50)
			#
			# # ensure at least some circles were found
			# if circles is not None:
			# 	# convert the (x, y) coordinates and radius of the circles to integers
			# 	circles = np.round(circles[0, :]).astype("int")
			#
			# 	# loop over the (x, y) coordinates and radius of the circles
			# 	for (x, y, r) in circles:
			# 		cv2.circle(cv_frame,(x, y),r,(0,255,0),1) # draw the outer circle
			# 		cv2.circle(cv_frame,(x, y),2,(0,0,255),3) # draw the center of the circle

			# show the frame to our screen
			cv2.imshow('detected circles', cv_frame)
			cv2.imshow('hsv', res)
			key = cv2.waitKey(5)
			#print_camera_information(zed)
	# Close the camera
	zed.close()

# Desc: Determine the expected number of pixels in a defined dimension ('x'
#       or 'y') for a standard tennis ball for the inputed distance and camera\
#       paramters
# Inputs: pyzed.camera zed, distance, dimension ('x' or 'y')
# Outputs: (pixels_x, pixels_y)
def size_tennis_ball_pixels(zed, distance):
	# get focal length in pixels of camera
	calibration_params = zed.get_camera_information().calibration_parameters

	focal_length_x = calibration_params.left_cam.fx
	focal_length_y = calibration_params.left_cam.fy

	if distance is not None and not np.isnan(distance) and not np.isinf(distance):
		pixels_x = math.ceil(TENNIS_BALL_DIAMETER / distance * focal_length_x)
		pixels_y = math.ceil(TENNIS_BALL_DIAMETER / distance * focal_length_y)
		return pixels_x, pixels_y
	else:
		return 0, 0
	# print("Expected number of pixels for tennis ball x: {0}, y: {1}".format(pixels_x, pixels_y))

# Desc: Determine the distance to the largest contour identified by it's moment
# Inputs: cv2.moment, pyzed.core.PyMat matrix= point_cloud_matrix
# Outputs: int distance
def distance_to_largest_contour(moment, point_cloud_matrix):
	############################################
	# Intermittent divide by 0 error!!
	############################################
	if moment["m00"] != 0:
		cx = int(moment['m10'] / moment['m00'])
		cy = int(moment['m01'] / moment['m00'])

		# Get and print distance value in mm at the center of the image
		# We measure the distance camera - object using Euclidean distance
		err, point_cloud_value = point_cloud_matrix.get_value(int(cx), int(cy))

		distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
		                     point_cloud_value[1] * point_cloud_value[1] +
		                     point_cloud_value[2] * point_cloud_value[2])

		if distance is not None and not np.isnan(distance) and not np.isinf(distance):
			distance = round(distance)
			print("Distance to Camera at ({0}, {1}): {2} mm\n".format(cx, cy, distance))
			sys.stdout.flush()
			return distance
		else:
			print("Can't estimate distance at this position, move the camera\n")
			sys.stdout.flush()
			return -1

def print_camera_information(camera):
	print("Resolution: {0}, {1}.".format(camera.get_resolution().width, camera.get_resolution().height))
	print("Camera FPS: {0}.".format(camera.get_camera_fps()))
	print("Firmware: {0}.".format(camera.get_camera_information().firmware_version))
	print("Serial number: {0}.\n".format(camera.get_camera_information().serial_number))

# def adjust_hsv_filter(key,)
# 	if key == ord('2'):
# 		h1 += 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('w'):
# 		h1 -= 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('s'):
# 		h2 += 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
# 	elif key == ord('x'):
# 		h2 -= 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
# 	elif key == ord('3'):
# 		s1 += 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('e'):
# 		s1 -= 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('d'):
# 		s2 += 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
# 	elif key == ord('c'):
# 		s2 -= 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
# 	elif key == ord('4'):
# 		v1 += 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('r'):
# 		v1 -= 1
# 		yellowLower = (h1, s1, v1)
# 		print("yellowLower = " + str(yellowLower))
# 	elif key == ord('f'):
# 		v2 += 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
# 	elif key == ord('v'):
# 		v2 -= 1
# 		yellowUpper = (h2, s2, v2)
# 		print("yellowUpper = " + str(yellowUpper))
if __name__ == "__main__":
	main()