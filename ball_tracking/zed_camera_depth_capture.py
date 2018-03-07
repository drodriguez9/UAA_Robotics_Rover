import cv2
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import math
import numpy as np
import sys

def main():
	# Create a PyZEDCamera object
	zed = zcam.PyZEDCamera()

	# Create a PyInitParameters object and set configuration parameters
	init_params = zcam.PyInitParameters()
	init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # Use PERFORMANCE depth mode
	init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # Use milliliter units (for depth measurements)

	# Open the camera
	err = zed.open(init_params)
	if err != tp.PyERROR_CODE.PySUCCESS:
		exit(1)

	# Create and set PyRuntimeParameters after opening the camera
	runtime_parameters = zcam.PyRuntimeParameters()
	runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode

	# Grab reference to webcam through OpenCV
	# camera = cv2.VideoCapture(0)

	# define the lower and upper boundaries of the "orange"
	# ball in the HSV color space, then initialize the
	# list of tracked points

	h1 = 35; h2 = 85
	s1 = 70; s2 = 255
	v1 = 30; v2 = 255

	yellowLower = (h1, s1, v1)
	yellowUpper = (h2, s2, v2)

	# Capture images and depth until 'q' is pressed
	left_image = core.PyMat()
	right_image = core.PyMat()
	depth = core.PyMat()
	point_cloud = core.PyMat()

	while True:
		# A new image is available if grab() returns PySUCCESS
		if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
			# Retrieve left image
			zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
			# Retrieve right image
			zed.retrieve_image(right_image, sl.PyVIEW.PyVIEW_RIGHT)
			# Retrieve depth map. Depth is aligned on the left image
			zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
			# Retrieve colored point cloud. Point cloud is aligned on the left image.
			zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)

			# Get and print distance value in mm at the center of the image
			# We measure the distance camera - object using Euclidean distance
			x = round(left_image.get_width() / 2)
			y = round(left_image.get_height() / 2)
			err, point_cloud_value = point_cloud.get_value(x, y)

			distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
								 point_cloud_value[1] * point_cloud_value[1] +
								 point_cloud_value[2] * point_cloud_value[2])

			if not np.isnan(distance) and not np.isinf(distance):
				distance = round(distance)
				print("Distance to Camera at ({0}, {1}): {2} mm\n".format(x, y, distance))
			else:
				print("Can't estimate distance at this position, move the camera\n")
			sys.stdout.flush()

			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break
	# Close the camera
	zed.close()

if __name__ == "__main__":
	main()