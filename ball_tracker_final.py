##############################################
# Follow instructions here to get OpenCV3 (cv2)
# to work with python3:
# https://jkjung-avt.github.io/opencv3-on-tx2/
# And Step 5 from: 
# https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
#####################################

import cv2
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import math
import numpy as np
import sys

TENNIS_BALL_DIAMETER = 68.6 #millimeters

def main():
	# Define the lower and upper boundaries of "yellow" in HSV color space
	# and default the iterations of the erode and dilate functions.

	h1 = 35; h2 = 85
	s1 = 70; s2 = 255
	v1 = 30; v2 = 255
	yellow_hsv_lower = np.array([h1, s1, v1])
	yellow_hsv_upper = np.array([h2, s2, v2])
	erosion_iterations = 2
	dilation_iterations = 3
	print_help()

	run_program = True
	while run_program:
		key = input("Enter command:")
		if key == 'q':
			print("\nExiting program...")
			run_program = False
		elif key == 'h':
			print_help()
		elif key == 'r':
			find_tennis_ball(hsv_lower=yellow_hsv_lower, hsv_upper=yellow_hsv_upper,
			                 num_erosions=erosion_iterations, num_dilations=dilation_iterations)
			print_help()
		elif key == 'v':
			yellow_hsv_lower, yellow_hsv_upper, erosion_iterations, dilation_iterations = \
				adjust_hsv_filter(hsv_lower=yellow_hsv_lower, hsv_higher=yellow_hsv_upper,
				                  num_erosions=erosion_iterations, num_dilations=dilation_iterations)
			print_help()

# Desc: Uses the ZED SDK and OpenCV libraries to identify a circlular object of
#       a specified color through computer vision. Current settings of the ZED
#       camera can detect a tennis ball 0.7 meters away from the left lens. This
#       minimum detection distance can be reduced to 0.3 meters, but at high
#       computation cost.
#
#       The function uses the inputed arguments to create a mask for the color
#       specified. It uses that mask to make a guess where the circular object is
#       on screen. Then a Region of Interest (ROI) around the color is passed on
#       to the circle Hough Transform. If a circle is detected, the function
#       assumes the circle is target and provides rover coordinates for the object.
#
#       If a circle is not detected, the function will provide coordinates for the
#       color region detected as a guess for the rover to investigate.
#
#       hsv_lower is an array with 3 indices (Hue, Saturation, Value/Brightness)
#       hsv_higher is an array with 3 indices (Hue, Saturation, Value/Brightness)
#       num_erosions is how many iterations of the OpenCV erode function
#       num_dilations is how many iterations of the OpenCV dilate function
#
# Inputs: int[2] hsv_lower, int[2] hsv_higher, int num_erosions, int num_dilations
# Outputs: int[2] hsv_lower, int[2] hsv_higher, int num_erosions, int num_dilations
def find_tennis_ball(hsv_lower, hsv_upper, num_erosions, num_dilations):
	# Create a PyZEDCamera object
	zed = zcam.PyZEDCamera()

	# Create a PyInitParameters object and set configuration parameters
	init_params = zcam.PyInitParameters()
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

	# Initialize arrays with proper format from ZED SDK
	left_image = core.PyMat()
	point_cloud = core.PyMat()

	# Update user with information
	print_camera_information(camera=zed)

	# Capture images and depth until 'q' is pressed
	key = ''
	while key != 113:  # for 'q' key
		# grab the current frame
		if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
			# Retrieve left image
			zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
			cv_frame = left_image.get_data()

			# Retrieve colored point cloud. Point cloud is aligned on the left camera
			zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZ)

			# Resize the frame, blur it, and convert it to the HSV color space
			blur_frame = cv2.medianBlur(cv_frame, 1)
			hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

			# Construct a mask for the color "yellow", then perform a series of dilations
			# and erosions to remove any small blobs left in the mask
			# OpenCV examples found here: https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html
			# More information on morphological operations:
			#   https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
			mask1 = cv2.inRange(hsv, hsv_lower, hsv_upper)
			mask2 = cv2.erode(mask1, None, iterations=num_erosions)
			mask3 = cv2.dilate(mask2, None, iterations=num_dilations)

			# Find contours in the mask
			# Documentation for findContours found here:
			# https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
			# findCountours returns an array of x,y coordinates that outlines the contours
			cnts = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

			guess_x_coord = 0
			guess_y_coord = 0
			guess_radius = 0

			# only proceed if at least one contour was found
			if len(cnts) > 0:
				# find the largest contour in mask3
				c = max(cnts, key=cv2.contourArea)

				#################################
				# For debugging
				#
				# cv2.drawContours(cv_frame, c, -1, (0, 255, 0), 3)
				#################################

				# guess where the tennis ball is from the largest contour,
				# draw a yellow circle around it, and find the x, y coordinates
				guess_x_coord, guess_y_coord, guess_radius = guess_circle(contour_array=c)

				# determine centroid coordinates and distance to centroid from camera
				distance_to_ball, centroid_x_coord, centroid_y_coord, tennis_ball_coord = \
					coordinates_to_largest_contour(moment=cv2.moments(c), point_cloud_matrix=point_cloud)

				if distance_to_ball != -1: #if coordinates_to_largest_contour did not error
					tennis_ball_pixels_x, tennis_ball_pixels_y = \
						size_tennis_ball_pixels(zed=zed, distance=distance_to_ball)

					# Can't have non-integer pixels, so take the ceiling of the pixels/2 to create a Region of Interest (ROI)
					# Also, add some buffer pixels to deal with image rectication stretching the image (mostly around
					# the edges).
					buffer = tennis_ball_pixels_x * 0.5
					x1 = math.ceil(centroid_x_coord - (tennis_ball_pixels_x / 2 + buffer))
					x2 = math.ceil(centroid_x_coord + (tennis_ball_pixels_x / 2 + buffer))
					y1 = math.ceil(centroid_y_coord - (tennis_ball_pixels_y / 2 + buffer))
					y2 = math.ceil(centroid_y_coord + (tennis_ball_pixels_y / 2 + buffer))
					image_width = left_image.get_width()
					image_height = left_image.get_height()

					# If ROI extends off the frame, program crashes. Checks if the ROI would extend
					# off the frame. If so, the code does not run.
					#
					###############################################################################
					# FUTURE WORK:
					#   Instead of not running the following code when the ROI extends off the
					#   frame, shift/change the ROI to only go to the edge of the frame.
					###############################################################################
					if (0 <= y1 <= image_height and 0 <= y2 <= image_height and
							0 <= x1 <= image_width and 0 <= x2 <= image_width and
							x1 != x2 and y1 != y2):
						tennis_ball = cv_frame[y1:y2, x1:x2]

						#################################
						# For debugging
						#
						# cv2.rectangle(img=cv_frame, pt1=(x1, y1), pt2=(x2,y2), color=(0,0,255), thickness=2)
						#
						#################################

						# expected radius from averaging tennis_ball_pixels_x & tennis_ball_pixels_y
						radius = int(((tennis_ball_pixels_x + tennis_ball_pixels_y) / 2) / 2)
						circles = detect_circle(frame=tennis_ball, expected_radius=radius)

						# ensure at least some circles were found
						if circles is not None:
							# convert the (x, y) coordinates and radius of the circles to integers
							circles = np.round(circles[0, :]).astype("int")

							# draw the largest circle (the 0th index of array circles) by x-y coordinates and radius
							x = circles.item(0, 0)
							y = circles.item(0, 1)
							r = circles.item(0, 2)
							cv2.circle(img=tennis_ball, center=(x, y), radius=r, color=(0, 0, 255), thickness=2)
							distance, azimuth, elevation = get_rover_coordinates(x_coord=tennis_ball_coord[0],
							                                                     y_coord=tennis_ball_coord[1],
							                                                     z_coord=tennis_ball_coord[2])
							#################################
							# For debugging
							#
							print("Detected tennis ball at\n"
							      "    Distance: {0} meters\n"
							      "    Rotate: {1} degrees\n"
							      "    Elevate: {2} degrees\n".format(np.around(distance / 1000, decimals=2),
							                                          np.around(azimuth * 180 / np.pi, decimals=2),
							                                          np.around(elevation * 180 / np.pi, decimals=2)))
							#
							#################################
						else:
							# If a circle is not detected, guess the rover coordinates by color only
							distance_guess, azimuth_guess, elevation_guess = get_rover_coordinates(
								x_coord=tennis_ball_coord[0],
								y_coord=tennis_ball_coord[1],
								z_coord=tennis_ball_coord[2])
							#################################
							# For debugging
							#
							print("Guessed tennis ball location at\n"
							      "    Distance: {0} meters\n"
							      "    Rotate: {1} degrees\n"
							      "    Elevate: {2} degrees\n".format(np.around(distance_guess / 1000, decimals=2),
							                                          np.around(azimuth_guess * 180 / np.pi, decimals=2),
							                                          np.around(elevation_guess * 180 / np.pi, decimals=2)))
							#
							#################################

			if guess_radius > 1:
				# draw the guess_circle on the frame,
				cv2.circle(img=cv_frame, center=(int(guess_x_coord), int(guess_y_coord)), radius=int(guess_radius)
						   , color=(0, 255, 255), thickness=1)
			cv2.drawMarker(img=cv_frame, position=(guess_x_coord, guess_y_coord), color=(0, 0, 255),
						   markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)

			# show the frame to our screen
			cv2.imshow('final', cv_frame)
			# cv2.imshow('temp', cv_frame_temp)
			# cv2.imshow('mask', mask3)

			key = cv2.waitKey(5)
	# Close the camera
	cv2.destroyAllWindows()
	zed.close()
	print("Exiting tennis ball detection module...")

# Desc: Transform the ZED x,y,z-coordinate system to distance, elevation, and azimuth (rotation)
#           Distance is always positive
#           Elevation uses x-z plane as 0 radian, with -y direction as "+"/up and
#               y direction as "-"/down
#           Rotation is around the y-axis; starts at 0 radians at the z-axis and rotates
#               in the "+" direction towards the x-axis/ "-" direction towards the -x-axis
#       Documentation for ZED coordinate system:
#       https://docs.stereolabs.com/overview/positional-tracking/coordinate-frames/
# Inputs: float x_coord, float y_coord, float z_coord
# Outputs: int r, int y, int radius
def get_rover_coordinates(x_coord, y_coord, z_coord):
	hxz = np.hypot(x_coord, z_coord)
	# r is distance from left lens to coordinate
	r = np.hypot(hxz, y_coord)
	# elevation is "pitch" or angle (in radians) of rotation up ("+") or down ("-")
	elevation = -np.arctan2(y_coord, hxz)
	# azimuth is "yaw" or angle (in radians) of rotation left ("-") or right ("+")
	azimuth = np.arctan2(x_coord, z_coord)
	return float(r), float(elevation), float(azimuth)

# Desc: Use minEnclosingCirce function to guess where circle is and return the circle
#       center point and radius.
#       Documentation for minEnlcosingCircle found here:
#       https://docs.opencv.org/3.4.1/d3/dc0/group__imgproc__shape.html#ga8ce13c24081bbc7151e9326f412190f1
# Inputs: cv2 frame, array contour_area
# Outputs: int x, int y, int radius
def guess_circle(contour_array):
	((x, y), radius) = cv2.minEnclosingCircle(contour_array)
	return int(x), int(y), int(radius)

# Desc: Use HoughCircles function to find the tennis ball by its circular shape.
#       Documentation for HoughCircles found here:
#       https://docs.opencv.org/3.4.1/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
#       A great summary of the function is found in the second comment here:
#       https://dsp.stackexchange.com/questions/22648/in-opecv-function-hough-circles-how-does-parameter-1-and-2-affect-circle-detecti
# Inputs: cv2 frame, int expected_radius
# Outputs: array of circles[index_of_circle][x_coord][y_coord][radius]
def detect_circle(frame, expected_radius):
	blur_frame = cv2.medianBlur(frame, 1)
	gray_frame = cv2.cvtColor(blur_frame,cv2.COLOR_BGR2GRAY)

	# setting minDist to 1 because ASSUMING (ass U me) there is only 1 tennis ball
	min_dist = 1
	circles = cv2.HoughCircles(image=gray_frame,method=cv2.HOUGH_GRADIENT,dp=1,
							   minDist=min_dist,param1=200,param2=15,
							   minRadius=math.floor(expected_radius*.8),
							   maxRadius=math.ceil(expected_radius*1.2))
	#################################
	# For debugging
	#
	# print('Expected radius between {0} - {1})\n'.format(int(expected_radius*.8), int(expected_radius*1.2)))
	# cv2.imshow('gray', gray_frame)
	#################################
	return circles

# Desc: Calculate the expected number of pixels in the x & y dimension
#       for a standard tennis ball from the inputed distance and focal length.
#       The focal length is pulled directly from the ZED camera parameters.
# Inputs: pyzed.camera zed, int distance
# Outputs: (pixels_x, pixels_y)
def size_tennis_ball_pixels(zed, distance):
	# get focal length in pixels of camera
	calibration_params = zed.get_camera_information().calibration_parameters

	focal_length_x = calibration_params.left_cam.fx
	focal_length_y = calibration_params.left_cam.fy

	# somehow distance can be None (Null); using conditional logic to prevent crash
	if distance is not None and not np.isnan(distance) and not np.isinf(distance):
		pixels_x = math.ceil(TENNIS_BALL_DIAMETER / distance * focal_length_x)
		pixels_y = math.ceil(TENNIS_BALL_DIAMETER / distance * focal_length_y)
		#################################
		# For debugging
		#
		# print("Expected pixel dimensions for Tennis Ball on image\n"
		#      "    x-dimension: {0} pixels\n"
		#      "    y-dimension: {1} pixels\n".format(pixels_x, pixels_y))
		#
		#################################
		return pixels_x, pixels_y
	else:
		return 0, 0

# Desc: Determine the distance to the largest contour identified by it's moment
#       and provide the centroid's x and y coordinate, along with the point cloud
#       data (x, y, z in millimeters from left ZED lens) for that location.
#       If no distance is identified return -1 (impossible distance).
#       Reference for image moments: https://en.wikipedia.org/wiki/Image_moment
#       m00 = area
# Inputs: cv2.moment, pyzed.core.PyMat matrix= point_cloud_matrix
# Outputs: int distance, int cx (centroid_x_coord), int cy (centroid_y_coord), point_cloud_data
def coordinates_to_largest_contour(moment, point_cloud_matrix):
	#if moment["m00"] != 0:
	centroid_x = int(moment['m10'] / moment['m00'])
	centroid_y = int(moment['m01'] / moment['m00'])

	# Get and print distance value in mm at the center of the image
	# We measure the distance camera - object using Euclidean distance
	err, point_cloud_value = point_cloud_matrix.get_value(int(centroid_x), int(centroid_y))

	###############################################################################
	# FUTURE WORK:
	#   Intermitentmly, the point_cloud_matrix.get_value() returns err = 'success'
	#   and point_cloud_value = 'Nan'. That is weird... and I guess it is coming
	#   from the ZED SDK. Figuring out why could help remove the extra conditional
	#   logic below with "if distance is not None..."
	###############################################################################
	distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
						 point_cloud_value[1] * point_cloud_value[1] +
						 point_cloud_value[2] * point_cloud_value[2])

	if distance is not None and not np.isnan(distance) and not np.isinf(distance):
		distance = round(distance)
		#################################
		# For debugging
		#
		# print("Distance from Tennis Ball to Camera at ({0}, {1}): {2} mm\n".format(centroid_x, centroid_y, distance))
		#
		#################################
		sys.stdout.flush()
		return distance, centroid_x, centroid_y, point_cloud_value
	else:
		#################################
		# For debugging
		#
		# print("Can't estimate distance at this position, move the camera\n")
		#
		#################################

		sys.stdout.flush()
		return -1, None, None, None

# Desc: Helper function to print ZED camera information
# Inputs:
# Outputs:
def print_camera_information(camera):
	print("Resolution: {0}, {1}.".format(camera.get_resolution().width, camera.get_resolution().height))
	print("Camera FPS: {0}.".format(camera.get_camera_fps()))
	print("Firmware: {0}.".format(camera.get_camera_information().firmware_version))
	print("Serial number: {0}.\n".format(camera.get_camera_information().serial_number))
	sys.stdout.flush()

# Desc: Helper function to print options for main menu
# Inputs:
# Outputs:
def print_help():
	print("\n*Recommend adjusting HSV filter before running detection module.*\n"
		  "Help for tennis ball detection program\n"
		  "  Run tennis ball detection module:      r\n"
		  "  Adjust hsv filter for tennis ball:     v\n"
		  "  Show this help again:                  h\n"
		  "  Quit:                                  q\n")

# Desc: Helper function to print options for adjusting hsv filter
# Inputs:
# Outputs:
def print_help_hsv():
	print("\n Controls for adjusting the HSV filter:\n"
		  "    Select 'minimum hue':             1\n"
		  "    Select 'minimum saturation':      2\n"
		  "    Select 'minimum brightness':      3\n"
		  "    Select 'maximum hue':             4\n"
		  "    Select 'maximum saturation':      5\n"
		  "    Select 'maximum brightness':      6\n"
		  "    Select 'number of erosions:       7\n"
	      "    Select 'number of dilations:      8\n"
		  "    Increase hsv settings value:      +\n"
		  "    Decrease hsv settings value:      -\n"
		  "    Cycle through the HSV masks:      m\n"
		  "    Display current HSV filter:       s\n"
	      "    Reset to original settings:       r\n"
		  "    Exit adjusting HSV filter:        q\n"
		  "    Show this information again:      h")

# Desc: Helper function to check if input is negative and inform the
#       user through CLI
# Inputs: int integer
# Outputs: boolean T/F
def is_negative_number(integer):
	if integer < 0:
		print("Can't set to a negative number.")
		return True
	else:
		return False

# Desc: Allows user to adjust the OpenCV settings for identifying the
#       tennis ball by color. On exit, the user is given the option to save
#       the changes they made. If not, the initial values are returned.
#       hsv_lower is an array with 3 indices (Hue, Saturation, Value/Brightness)
#       hsv_higher is an array with 3 indices (Hue, Saturation, Value/Brightness)
#       num_erosions is how many iterations of the OpenCV erode function
#       num_dilations is how many iterations of the OpenCV dilate function
# Inputs: int[2] hsv_lower, int[2] hsv_higher, int num_erosions, int num_dilations
# Outputs: int[2] hsv_lower, int[2] hsv_higher, int num_erosions, int num_dilations
def adjust_hsv_filter(hsv_lower, hsv_higher, num_erosions, num_dilations):
	# Create a PyZEDCamera object
	zed = zcam.PyZEDCamera()

	# Create a PyInitParameters object and set configuration parameters
	init_params = zcam.PyInitParameters()
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

	# Initialize arrays with proper format from ZED SDK
	left_image = core.PyMat()

	# Update user with information
	print_camera_information(camera=zed)

	print("Adjusting HSV filter for tennis ball detection."
		  "\nHere you can adjust the filter while viewing the mask live."
		  "For best results, ensure the hsv filter is adjusted every time "
		  "before running the tennis ball detection program.")
	print_help_hsv()

	# Set parameters from function arguements
	hsv_temp = np.array([hsv_lower, hsv_higher])
	num_erosions_temp = num_erosions
	num_dilations_temp = num_dilations
	setting_hsv_low_or_high = 0
	setting_hue_sat_val = 0
	setting_erode = False
	setting_dilate = False
	str_setting = "No setting selected"
	key = ''
	mask_cycle = 1
	run_hsv_filter = True

	# Capture images until 'q' is pressed
	while run_hsv_filter:
		# grab the current frame
		if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
			# Retrieve left image
			zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
			cv_frame = left_image.get_data()

			# Blur the frame and convert it to the HSV color space
			blur_frame = cv2.medianBlur(cv_frame, 1)
			hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

			# Construct a mask for the color "yellow", then perform a series of dilations
			# and erosions to remove any small blobs left in the mask
			# OpenCV examples found here: https://docs.opencv.org/master/db/df6/tutorial_erosion_dilatation.html
			# More information on morphological operations:
			#   https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
			mask1 = cv2.inRange(hsv, hsv_temp[0], hsv_temp[1])
			mask2 = cv2.erode(mask1, None, iterations=num_erosions_temp)
			mask3 = cv2.dilate(mask2, None, iterations=num_dilations_temp)

			# When q is pressed, the entire function will end from inside the nested loops,
			# returning either the old or new HSV filter parameters.
			if key == ord('q'):
				while True:
					# Close the camera
					cv2.destroyAllWindows()
					zed.close()
					temp = input("\nDo you want to save the new hsv filter? [Y/N]:")
					if temp == 'Y':
						return hsv_temp[0], hsv_temp[1], num_erosions_temp, num_dilations_temp
					elif temp == 'N':
						return hsv_lower, hsv_higher, num_erosions, num_dilations
			elif key == ord('1'):
				setting_hsv_low_or_high = 0 # set to LOW
				setting_hue_sat_val = 0     # set to HUE
				setting_erode = False
				setting_dilate = False
				str_setting = "Minimum hue"
				print("\nHSV setting: " + str_setting)
			elif key == ord('2'):
				setting_hsv_low_or_high = 0 # set to LOW
				setting_hue_sat_val = 1     # set to SATURATION
				setting_erode = False
				setting_dilate = False
				str_setting = "Minimum saturation"
				print("\nHSV setting: " + str_setting)
			elif key == ord('3'):
				setting_hsv_low_or_high = 0 # set to LOW
				setting_hue_sat_val = 2     # set to VALUE
				setting_erode = False
				setting_dilate = False
				str_setting = "Minimum brightness"
				print("\nHSV setting: " + str_setting)
			elif key == ord('4'):
				setting_hsv_low_or_high = 1 # set to HIGH
				setting_hue_sat_val = 0     # set to HUE
				setting_erode = False
				setting_dilate = False
				str_setting = "maximum hue"
				print("\nHSV setting: " + str_setting)
			elif key == ord('5'):
				setting_hsv_low_or_high = 1 # set to HIGH
				setting_hue_sat_val = 1     # set to SATURATION
				setting_erode = False
				setting_dilate = False
				str_setting = "Maximum saturation"
				print("\nHSV setting: " + str_setting)
			elif key == ord('6'):
				setting_hsv_low_or_high = 1 # set to HIGH
				setting_hue_sat_val = 2     # set to VALUE
				setting_erode = False
				setting_dilate = False
				str_setting = "Maximum brightness"
				print("\nHSV setting: " + str_setting)
			elif key == ord('7'):
				setting_erode = True # set to ERODE
				setting_dilate = False
				str_setting = "Number of erosions"
				print("\nHSV setting: " + str_setting)
			elif key == ord('8'):
				setting_erode = False
				setting_dilate = True # set to DILATE
				str_setting = "Number of dilations"
				print("\nHSV setting: " + str_setting)

			elif key == ord('+') or key == 171: # includes numpad +
				if setting_dilate == False and setting_erode == False:
					hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val] += 1
					print(str_setting + ": " + str(hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val]))
					if setting_hsv_low_or_high == 1:
						print("HSV high setting [Hue, Saturation, Brightness] = " + str(hsv_temp[1]))
					elif setting_hsv_low_or_high == 0:
						print("HSV low setting [Hue, Saturation, Brightness] = " + str(hsv_temp[0]))
				elif setting_erode == True:
					num_erosions_temp += 1
					print(str_setting + ": " + str(num_erosions_temp))
				elif setting_dilate == True:
					num_dilations_temp += 1
					print(str_setting + ": " + str(num_dilations_temp))
			elif key == ord('-') or key == 173: # includes numpad -
				if setting_dilate == False and setting_erode == False:
					hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val] -= 1
					if is_negative_number(hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val]):
						hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val] += 1
					print(str_setting + ": " + str(hsv_temp[setting_hsv_low_or_high, setting_hue_sat_val]))
					if setting_hsv_low_or_high == 1:
						print("HSV high setting [Hue, Saturation, Brightness] = " + str(hsv_temp[1]))
					elif setting_hsv_low_or_high == 0:
						print("HSV low setting [Hue, Saturation, Brightness] = " + str(hsv_temp[0]))
				elif setting_erode == True:
					num_erosions_temp -= 1
					if is_negative_number(num_erosions_temp):
						num_erosions_temp += 1
					print(str_setting + ": " + str(num_erosions_temp))
				elif setting_dilate == True:
					num_dilations_temp -= 1
					if is_negative_number(num_dilations_temp):
						num_dilations_temp += 1
					print(str_setting + ": " + str(num_dilations_temp))
			elif key == ord('s'):
				print("\n Current HSV filter: ({0}, {1})".format(hsv_temp[0], hsv_temp[1]))
			elif key == ord('r'):
				hsv_temp = np.array([hsv_lower, hsv_higher])
				num_erosions_temp = num_erosions
				num_dilations_temp = num_dilations
				setting_hsv_low_or_high = 0
				setting_hue_sat_val = 0
				setting_erode = False
				setting_dilate = False
				str_setting = "No setting selected"
				mask_cycle = 1
				print("\n Resetting settings to orginal values...\n"
				      "    HSV filter: ({0}, {1})\n"
				      "    Number of erosions: {2}\n"
				      "    Number of dilations: {2}\n".format(
					hsv_temp[0], hsv_temp[1], num_erosions_temp, num_dilations_temp))
				cv2.destroyAllWindows()
			elif key == ord('h'):
				print_help_hsv()
			elif key == ord('m'):
				if mask_cycle == 1:
					mask_cycle = 2
					print("mask_cycle = HSV Filter with Erosions")
					cv2.destroyAllWindows()
				elif mask_cycle == 2:
					mask_cycle = 3
					print("mask_cycle = HSV Filter with Erosions + Dilations")
					cv2.destroyAllWindows()
				elif mask_cycle == 3:
					mask_cycle = 1
					print("mask_cycle = RAW HSV Filter")
					cv2.destroyAllWindows()

			if mask_cycle == 1:
				cv2.imshow("RAW HSV Mask", mask1)
			elif mask_cycle == 2:
				cv2.imshow("HSV Mask + Erosions", mask2)
			else:
				cv2.imshow("HSV Mask + Erosions + Dilations", mask3)

			key = cv2.waitKey(5)

if __name__ == "__main__":
	main()
