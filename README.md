# sidewalk_detector

This package contains a ROS node named `sidewalk_detector`. It subscribes to RGB and Depth image topics, as well as PointCloud2 topics, typically from an RGB-D sensor. The node processes this data to identify the sidewalk and publishes the results.

## Published Topics

The `sidewalk_detector` node publishes the following topics:

1.  `/sidewalk_detector/color/image_raw`: Outputs the original RGB image with a visual highlight (e.g., a red mask) overlaid on the pixels classified as belonging to the sidewalk.
2.  `/sidewalk_detector/depth/points_in`: Outputs a `sensor_msgs/PointCloud2` message containing the subset of points from the input point cloud (e.g., from `/camera/depth/points`) that are considered to be part of the sidewalk plane (inliers from RANSAC segmentation).
3.  `/sidewalk_detector/depth/points_out`: Outputs a `sensor_msgs/PointCloud2` message containing the subset of points from the input point cloud that are considered to be outside the sidewalk plane (outliers from RANSAC segmentation and points filtered by the Passthrough filter).

## Installation

1.  Clone this repository into your catkin workspace's `src` directory.
    ```bash
    cd your_catkin_ws/src
    git clone <repository_url>
    ```
2.  Build the workspace:
    ```bash
    cd ../.. 
    catkin_make
    ```
3.  Source the workspace:
    ```bash
    source devel/setup.bash
    ```

## Running the Node

To run the `sidewalk_detector` node:

```bash
rosrun sidewalk_detector sidewalk_detector
```

Ensure that your RGB-D sensor is publishing on the appropriate topics (default: `/camera/color/image_raw`, `/camera/depth/image_raw`, and `/camera/depth/points`).

## Approach

### General Assumption
The primary assumption is that the area immediately in front of the sensor is predominantly sidewalk. While the algorithm can handle some obstacles, the sensor should be positioned to have a clear view of the sidewalk.

### Point Cloud Processing (`pointCb`)

1.  **Input**: Raw `sensor_msgs/PointCloud2` from the sensor.
2.  **Passthrough Filter**: A PCL Passthrough filter is applied to the Y-axis to retain points within a specific height range relative to the sensor, effectively removing points far above or below the expected sidewalk level.
3.  **RANSAC Plane Segmentation**: PCL's RANSAC (Random Sample Consensus) algorithm is used on the filtered point cloud to identify the dominant planar surface, which is assumed to be the sidewalk.
4.  **Output**:
    *   Inlier points (those fitting the detected plane) are published to `/sidewalk_detector/depth/points_in`.
    *   Outlier points (those not fitting the plane, plus those initially removed by the Passthrough filter) are published to `/sidewalk_detector/depth/points_out`.

### Image Processing (`imageCb`)

1.  **Input**: Synchronized RGB (`sensor_msgs/Image`) and Depth (`sensor_msgs/Image`) messages.
2.  **Image Preprocessing**:
    *   Both RGB and Depth images are flipped vertically (adjusting for typical camera mounting).
    *   The Depth image is resized to a fixed resolution (e.g., 640x480) and normalized to an 8-bit grayscale representation.
3.  **Color-Based Analysis (RGB Image)**:
    *   A specific Region of Interest (ROI) is defined in the RGB image. This ROI is assumed to be a reliable sample of the sidewalk's appearance.
    *   The mean and standard deviation of color values (BGR channels) are calculated within this ROI.
4.  **Pixel-wise Classification**:
    *   Each pixel in the (flipped) RGB image is evaluated.
    *   A pixel is classified as part of the sidewalk if:
        *   Its BGR color values fall within a range defined by the ROI's mean color +/- a multiplier of the ROI's standard deviation for each channel.
        *   Its corresponding normalized depth value (from the processed depth image) is below a predefined threshold (i.e., it's close enough to the sensor).
5.  **Output**: The original (flipped) RGB image is published to `/sidewalk_detector/color/image_raw`, with pixels classified as sidewalk highlighted (typically in red).

### Visualization Tip
If using a `.bag` file that does not contain TF (transform) data, you can use `static_transform_publisher` to create a static transform between your fixed frame (e.g., `map` or `odom`) and the camera's depth optical frame. For example, if the camera is ~21 inches off the ground and tilted down:
```bash
rosrun tf static_transform_publisher 0 0 0.5334 0 1.57 0 map camera_depth_optical_frame 100 
```
(Adjust translation and rotation values as per your setup.) This helps in visualizing the point clouds correctly in RViz.

## Configuration

Key parameters for the sidewalk detection algorithm are defined as `static const` members within the `SideWalkDetector` class in `src/sidewalk_detector.cpp`. These include:

*   **Image Processing**: ROI coordinates (`ROI_RECT`), depth threshold for sidewalk classification (`MAX_DEPTH_FOR_SIDEWALK`), starting row for classification (`CLASSIFICATION_START_ROW`), and standard deviation multipliers for color channels (`STDEV_MULT_B`, `STDEV_MULT_G`, `STDEV_MULT_R`).
*   **Point Cloud Processing**: Passthrough filter limits (`PASSTHROUGH_Y_MIN`, `PASSTHROUGH_Y_MAX`) and RANSAC distance threshold (`RANSAC_DISTANCE_THRESHOLD`).

If you need to fine-tune the detection behavior for different environments or sensor setups, these constants are the primary values to adjust.

## Code Structure

The main logic is encapsulated within the `SideWalkDetector` class (`src/sidewalk_detector.cpp`):

*   **`SideWalkDetector` Class**:
    *   Manages ROS subscriptions (to raw sensor data) and publications (for processed data).
    *   Initializes ROS node handle and image transport.
    *   Uses `message_filters::Synchronizer` to ensure RGB and Depth images are processed as pairs.
*   **`imageCb(...)`**:
    *   ROS callback for synchronized RGB and Depth image messages.
    *   Orchestrates the image preprocessing, color analysis, and pixel-wise classification to detect sidewalks in the 2D image.
*   **`pointCb(...)`**:
    *   ROS callback for `sensor_msgs/PointCloud2` messages.
    *   Handles 3D point cloud processing, including Passthrough filtering and RANSAC plane segmentation to identify the sidewalk plane.
*   **Helper Methods**:
    *   `process_depth_image(...)`: Handles flipping, resizing, and normalization of raw depth images.
    *   `classify_sidewalk_pixels(...)`: Performs the core logic of classifying individual pixels based on color and depth criteria.

The `main()` function in `src/sidewalk_detector.cpp` initializes the ROS node and the `SideWalkDetector` object.
