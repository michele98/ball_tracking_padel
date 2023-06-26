# Scope

This project is part of my Master's Thesis in Artificial intelligence. It aims to develop a system to track a ball in the 2D image space of a Padel match video. First, a neural network using the `TrackNetV2` architecture is trained to detect and localize the ball in each frame. This is done in the `detection` package.

From these detections, parabolic trajectories are subsequently fitted to track the ball and obtain statistics about its ball motion. This is done in the `trajectories` package.

The training data will be made available soon.

# Requirements

To run the code of this project, the following packages are required. The versions listed here are the ones that were installed on the machine that was used for development.
All the code was tested with the latest version of each package in June 2023, so newer versions up to that date should work without issues.
Older versions were not tested, but they might work as well.

Detection part, in the `detection` package:
 - `numpy 1.21.5`
 - `pandas 1.4.4`
 - `matplotlib 3.5.1`
 - `pytorch 1.12.1`
 - `opencv-python 4.5.5.64`

Trajectory fitting, in the `trajectories` package:
 - `networkx 2.8.4`
 - `numba 0.56.4`

The Python version that was used is `3.9.7`.
