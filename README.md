README
Contact Information:
    Name: Brendan Stassel
    Email: stasy.x@gmail.com

As of May 5, 2018 the UAA Robotics Club Jetson TX2 is configured to run the ~/PyCharm/ball_tracker/ball_tracker_final.py program in a terminal using python 3 and in the virtual environment ‘cv’ using the PyCharm IDE. A soft-linked ball_tracker_Brendan.py is saved on the Desktop of the Jetson TX2. If the system changes, such as upgrading the ZED SDK, CUDA, JetPack for TX2, etc., it is possible that the program will no longer run.

If this happens, a fresh install of JetPack 3.2 SDK with Linux for Tegra (L4T) R28.2 WILL NOT allow the Jetson TX2 to run the program. OpenCV 3.4.0 must be recompiled manually because the OpenCV4Tegra (installed from the JetPack 3.2 SDK) only provides support for python 2.7 and the ZED SDK requires python 3. The instructions I followed to recompile OpenCV 3.4.0 on the Jetson TX2 are here: https://jkjung-avt.github.io/opencv3-on-tx2/. I then also installed the ZED SDK and the ZED Python wrapper from Stereolabs. This will allow the Jetson TX2 to run the program from the terminal using python 3. In order to create a virtual environment for PyCharm, I followed the instructions from https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/ and installed the libraries using python 3.4.  I would recommend not installing any python examples or extra modules, as they take up a lot of the Jetson TX2’s limited hard drive space, that the pyimagesearch website installs.

This is a lengthy process and requires knowledge of how both Linux and CMake works. The process takes 2-3 hours and requires a separate ubuntu 16.04 computer too (for the JetPack SDK). Good luck and get ready to learn a bunch of Linux and CMake debugging!
