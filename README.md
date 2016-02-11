# sidewalk_detector

This package contains a node called “sidewalk_detector” that subscribes to RGB-D topics from a sensor and publishes to the following topics:

(1) /sidewalk_detector/color/image_raw

(2) /sidewalk_detector/depth/points_in

(3) /sidewalk_detector/depth/points_out

Definitions: 

where (1) outputs the images from the topic "/camera/color/image_raw” with a visible highlight (e.g. a red mask) mapped over the set of pixels that are considered to be INSIDE the sidewalk

where (2) outputs a point cloud which contains the subset of points from the point cloud output by the topic "/camera/depth/points” that are considered to be INSIDE the sidewalk

where (3) outputs a point cloud which contains the subset of points from the point cloud output by the topic "/camera/depth/points” that are considered to be OUTSIDE the sidewalk

#Install

Clone the repository in your workspace and do catkin_make.

To run the node:
rosrun sidewalk_detector sidewalk_detector

# Approach 

###TIP: since the bagfile does not have tf, a static_transform_publisher with a tf between fixed_frame and camera_depth_optical_frame (z=21", roll=1.57 rad) helps visualizing the pcl in rviz.

Assumption: the area immediately in front of the SENSOR is a sidewalk area (obstacles present in this area can be easily factored out using depth data but the sensor has to be primarily on sidewalk and not road)

### PCL

Preprocess the input point cloud by filtering out the obvious outliers. Select a subset of points near the ground and use RANSAC plane fitting to determine the plane of sidewalk.
Determine and filter out the outliers from original PCL by testing against the obtained plane.

### Color

Calibrate depth to rgb and get registered depth data.
Filter depth image. Select a subset of points near the sensor and use RANSAC plane fitting to determine the plane of sidewalk.
Determine and filter out the outliers by testing against the obtained plane. Map all inlier pixels on to the RGB image, removing most outlier pixels from RGB image . Get color histogram for a patch in front of sensor and use it to filter out more background pixels (including road) from the RGB image. Mark the rest of pixels as sidewalk.

Since the depth image quality from realsense was not very consistant, plane fitting could not be achieved instead color histogram thresholding after clustering was used. Pixels from the depth image too far from the camera were filtered out.
