//Author: Harsh Pandya
//Project: Sidewalk detector using RGB-D data

/*
Camera frames from topic header: camera_depth_optical_frame
                                 camera_color_optical_frame
Using static_tf_publisher with "map" fixed frame 
*/

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <vector>
#include <sstream>
#include <string>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <iomanip>

//SideWalkDetector class with callback members for depth-RGB and pcl
/**
 * @brief Detects sidewalks using RGB-D data from a camera.
 * 
 * This class subscribes to synchronized RGB and depth images, and point cloud data.
 * It processes the depth data to identify planar surfaces (potential sidewalks) and
 * uses color information from the RGB image to refine the sidewalk detection.
 * The resulting processed RGB image with detected sidewalk highlighted is published.
 */
class SideWalkDetector
{
private:
  // Constants for image processing in imageCb
  static const cv::Rect ROI_RECT; ///< Region of Interest for calculating color statistics.
  static const uchar MAX_DEPTH_FOR_SIDEWALK = 40; ///< Maximum normalized depth value to be considered part of the sidewalk.
  static const int CLASSIFICATION_START_ROW = 220; ///< Starting row index for pixel classification in the image.
  static const double STDEV_MULT_B; ///< Standard deviation multiplier for the Blue channel in color-based classification.
  static const double STDEV_MULT_G; ///< Standard deviation multiplier for the Green channel in color-based classification.
  static const double STDEV_MULT_R; ///< Standard deviation multiplier for the Red channel in color-based classification.

  // Constants for point cloud processing in pointCb
  static const float PASSTHROUGH_Y_MIN; ///< Minimum Y-coordinate for the Passthrough filter.
  static const float PASSTHROUGH_Y_MAX; ///< Maximum Y-coordinate for the Passthrough filter.
  static const double RANSAC_DISTANCE_THRESHOLD; ///< Distance threshold for RANSAC plane segmentation.

  /**
   * @brief Processes a raw depth image.
   * 
   * This function takes a raw depth image (typically 32FC1), flips it vertically,
   * resizes it to a target size, and normalizes its values to an 8-bit range (0-255).
   * @param depth_ptr_in Pointer to the input cv_bridge depth image.
   * @param target_size The desired size for the output processed depth image.
   * @return cv::Mat The processed 8-bit depth image.
   */
  cv::Mat process_depth_image(const cv_bridge::CvImagePtr& depth_ptr_in, const cv::Size& target_size)
  {
    cv::Mat depth_image_flipped;
    cv::flip(depth_ptr_in->image, depth_image_flipped, -1);

    cv::Mat depth_image_resized;
    cv::resize(depth_image_flipped, depth_image_resized, target_size);

    cv::Mat depth_src_normalized;
    double min_val, max_val;
    cv::minMaxIdx(depth_image_resized, &min_val, &max_val);
    // Avoid division by zero if min_val == max_val
    if (max_val - min_val > std::numeric_limits<double>::epsilon()) {
        depth_image_resized.convertTo(depth_src_normalized, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    } else {
        // Handle case where all depth values are the same (e.g., set to 0 or max range)
        depth_image_resized.convertTo(depth_src_normalized, CV_8UC1, 1.0, 0); // or some other appropriate handling
    }
    return depth_src_normalized;
  }

  /**
   * @brief Classifies pixels as sidewalk or non-sidewalk based on color and depth.
   * 
   * Iterates through a region of the image, comparing pixel colors to statistics
   * derived from a Region of Interest (ROI) and checking corresponding depth values.
   * Pixels identified as sidewalk are marked in red on the output_image.
   * @param output_image The image on which to mark sidewalk pixels (modified in place).
   * @param color_src The source color image (flipped BGR).
   * @param depth_src_processed The processed 8-bit depth image.
   * @param roi_mean Mean color values (BGR) calculated from the ROI.
   * @param roi_stdev Standard deviation of color values (BGR) calculated from the ROI.
   */
  void classify_sidewalk_pixels(cv::Mat& output_image,
                                const cv::Mat& color_src,
                                const cv::Mat& depth_src_processed,
                                const cv::Scalar& roi_mean,
                                const cv::Scalar& roi_stdev)
  {
    // The loop iterates over rows from CLASSIFICATION_START_ROW to height and all columns.
    // 'color_src' is used for color information, 'depth_src_processed' for depth.
    for(int i = CLASSIFICATION_START_ROW; i < depth_src_processed.rows; i++) 
    {
      for(int j = 0; j < depth_src_processed.cols; j++) 
      {
        cv::Vec3b val = color_src.at<cv::Vec3b>(i,j); // Color from original (flipped) image
        // Condition checks if color is within a statistical range derived from ROI 
        // AND if the corresponding normalized depth pixel is less than MAX_DEPTH_FOR_SIDEWALK.
        if(val[0] > (roi_mean[0] - STDEV_MULT_B * roi_stdev[0]) && val[0] < (roi_mean[0] + STDEV_MULT_B * roi_stdev[0]) && 
           val[1] > (roi_mean[1] - STDEV_MULT_G * roi_stdev[1]) && val[1] < (roi_mean[1] + STDEV_MULT_G * roi_stdev[1]) && 
           val[2] > (roi_mean[2] - STDEV_MULT_R * roi_stdev[2]) && val[2] < (roi_mean[2] + STDEV_MULT_R * roi_stdev[2]) && 
           (uchar)depth_src_processed.at<uchar>(i,j) < MAX_DEPTH_FOR_SIDEWALK)     
        {
          // Mark detected sidewalk pixels in red on the output image
          output_image.at<cv::Vec3b>(i,j)[0] = 0;   // Blue
          output_image.at<cv::Vec3b>(i,j)[1] = 0;   // Green
          output_image.at<cv::Vec3b>(i,j)[2] = 255; // Red
        }
      }
    }
  }

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  typedef image_transport::SubscriberFilter ImageSubscriber;  

  ros::NodeHandle nh_;  ///< ROS NodeHandle.
  image_transport::ImageTransport it_; ///< ImageTransport for handling image publishing and subscribing.
  image_transport::Publisher rgb_pub_; ///< Publisher for the processed RGB image.
  ros::Subscriber point_sub_; ///< Subscriber for the raw point cloud data.
  ros::Publisher in_point_pub_; ///< Publisher for inlier points from RANSAC.
  ros::Publisher out_point_pub_; ///< Publisher for outlier points from RANSAC.
  ros::Publisher filter_point_pub_; ///< Publisher for points filtered by the Passthrough filter (currently unused but kept for potential future use).
  ImageSubscriber rgb_sub_; ///< Subscriber for the raw RGB image.
  ImageSubscriber depth_sub_; ///< Subscriber for the raw depth image.
  message_filters::Synchronizer<MySyncPolicy> sync_; ///< Synchronizer for RGB and depth images.
  

public:
  /**
   * @brief Constructor for the SideWalkDetector class.
   * 
   * Initializes ROS node handle, image transport, subscribers, and publishers.
   * Sets up synchronized callback for RGB and depth images.
   */
  SideWalkDetector()
    : it_(nh_), 
      rgb_sub_(it_, "/camera/color/image_raw", 1), 
      depth_sub_(it_, "/camera/depth/image_raw", 1), 
      sync_(MySyncPolicy(100), rgb_sub_, depth_sub_)
  {
    // Subscribers
    point_sub_= nh_.subscribe("/camera/depth/points", 1, &SideWalkDetector::pointCb, this);
    sync_.registerCallback(boost::bind(&SideWalkDetector::imageCb,this, _1, _2));        

    // Publishers 
    rgb_pub_ = it_.advertise("/sidewalk_detector/color/image_raw", 1);     
    in_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_in", 1);
    out_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_out", 1);
    filter_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_filtered", 1);
  }
  
  /**
   * @brief ROS callback function for processing PointCloud2 messages.
   * 
   * This function takes a raw PointCloud2 message, converts it to a PCL point cloud,
   * applies a Passthrough filter to remove points far above the ground, performs
   * RANSAC plane segmentation to find the dominant plane (assumed to be the sidewalk),
   * and then publishes the inlier and outlier point clouds.
   * @param msg ConstPtr to the input sensor_msgs::PointCloud2 message.
   */
  void pointCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);      ///< Points belonging to the detected plane (inliers).
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);     ///< Points not belonging to the detected plane (outliers).
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>); ///< Points after Passthrough filter.
    sensor_msgs::PointCloud2::Ptr point_in (new sensor_msgs::PointCloud2);    ///< ROS message for inlier points.
    sensor_msgs::PointCloud2::Ptr point_out (new sensor_msgs::PointCloud2);   ///< ROS message for outlier points.
    sensor_msgs::PointCloud2::Ptr point_filter (new sensor_msgs::PointCloud2);///< ROS message for Passthrough filtered points.
    pcl::fromROSMsg (*msg, *cloud);

    // Filter out points much above ground using a Passthrough filter on the Y-axis.
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (PASSTHROUGH_Y_MIN, PASSTHROUGH_Y_MAX);
    pass.filter (*cloud_filtered);
    // Publish the result of the passthrough filter (optional, can be enabled for debugging)
    // pcl::toROSMsg (*cloud_filtered, *point_filter);
    // filter_point_pub_.publish(point_filter); 

    // Invert the filter to get points *not* in the Y range (those above the limits)
    // These are added back to the outliers later if they were not part of the original cloud_filtered.
    pass.setFilterLimitsNegative (true);
    pass.filter (*cloud); // 'cloud' now contains points outside the -1.0 to -0.3 Y range.

    // RANSAC plane segmentation to find the dominant plane in the filtered cloud.
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true); // Optional: refine coefficients.
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold (RANSAC_DISTANCE_THRESHOLD);

    seg.setInputCloud(cloud_filtered); // Use the Y-filtered cloud for plane segmentation.
    seg.segment (*inliers, *coefficients); // Perform segmentation.

    if (inliers->indices.empty())
    {
      //ROS_WARN("Could not estimate a planar model for the given dataset.");
      // If no plane is found, publish empty clouds or handle as appropriate
      // For now, we'll just convert empty clouds and publish.
      pcl::toROSMsg (*cloud_in, *point_in); // Will be empty
      pcl::toROSMsg (*cloud_out, *point_out); // Will be empty (or contain original cloud_filtered + cloud if logic below is hit)
    }
    else
    {
      // Extract inliers (plane) from the filtered cloud.
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(cloud_filtered);
      extract.setIndices(inliers);
      extract.setNegative(false); // Get the inliers.
      extract.filter(*cloud_in);

      // Extract outliers from the filtered cloud.
      extract.setNegative (true); // Get the outliers.
      extract.filter(*cloud_out);
    }
    
    // Combine the RANSAC outliers (from cloud_filtered) with the points originally filtered out by the passthrough filter (now in 'cloud').
    // This ensures 'cloud_out' contains all non-sidewalk points.
    *cloud_out += *cloud; 

    // Convert PCL clouds to ROS messages.
    pcl::toROSMsg (*cloud_in, *point_in);
    pcl::toROSMsg (*cloud_out, *point_out);
  
    // Publish output point cloud streams.
    in_point_pub_.publish(point_in);
    out_point_pub_.publish(point_out);
  }

  /**
   * @brief ROS callback function for synchronized RGB and Depth images.
   * 
   * This function is called when a new pair of synchronized RGB and depth images
   * is received. It processes these images to detect the sidewalk.
   * The RGB image is flipped, and a Region of Interest (ROI) is used to calculate
   * color statistics. The depth image is processed (flipped, resized, normalized).
   * Pixels are then classified as sidewalk based on color and depth criteria.
   * The resulting image with the sidewalk highlighted is published.
   * @param rgb ConstPtr to the input sensor_msgs::Image (RGB).
   * @param depth ConstPtr to the input sensor_msgs::Image (Depth).
   */
  void imageCb(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth)
  {
    // std::stringstream filename; // For saving images, not used currently
    cv_bridge::CvImagePtr cv_ptr;    
    try
    {
      cv_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
    cv_bridge::CvImagePtr depth_ptr;    
    try
    {
      depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
        
    // static int count=0; // For saving images, not used currently
    // count++;
    // int row = cv_ptr->image.size().height; // Unused
    // int col = cv_ptr->image.size().width;  // Unused

    cv::Size target_image_size(640,480); // Define target size for consistent processing.   

    // Flip the color image vertically.
    // The camera image might be upside down depending on its mounting.
    cv::flip(cv_ptr->image, cv_ptr->image, -1);    
    
    // Process the depth image (flip, resize, normalize).
    cv::Mat depth_src = process_depth_image(depth_ptr, target_image_size);
    
    // Clone the flipped color image to be used as the base for output and source for ROI.
    cv::Mat op = cv_ptr->image.clone();  // Output image, starts as a copy of flipped color image.
    cv::Mat src = cv_ptr->image.clone(); // Source color image (flipped).
    
    // Define and extract the Region of Interest (ROI) from the source color image.
    // This ROI is used to calculate statistics for color-based sidewalk detection.
    cv::Mat roi = src(ROI_RECT);         
    
    // Calculate mean and standard deviation of colors within the ROI.
    cv::Scalar mean_roi_color, stdev_roi_color;
    cv::meanStdDev(roi, mean_roi_color, stdev_roi_color);
    
    // Classify sidewalk pixels based on color (from ROI stats) and depth information.
    // The 'op' image is modified in place to highlight detected sidewalk pixels.
    classify_sidewalk_pixels(op, src, depth_src, mean_roi_color, stdev_roi_color);

    // Publish the resulting RGB image with sidewalk highlighted.
    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, op).toImageMsg();
    rgb_pub_.publish(rgb_msg);
    
  }

};

/**
 * @brief Main function for the sidewalk_detector_node.
 * 
 * Initializes the ROS node and creates an instance of the SideWalkDetector class.
 * It then enters a loop, allowing ROS to process callbacks.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return int Returns 0 on successful execution.
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "sidewalk_detector");
  SideWalkDetector sd;

  ros::spin(); // Process callbacks until shutdown.

  return 0;
}

// Initialize static const members for imageCb
const cv::Rect SideWalkDetector::ROI_RECT = cv::Rect(180, 260, 280, 210);
const double SideWalkDetector::STDEV_MULT_B = 2.0;
const double SideWalkDetector::STDEV_MULT_G = 2.0;
const double SideWalkDetector::STDEV_MULT_R = 3.0;

// Initialize static const members for pointCb
const float SideWalkDetector::PASSTHROUGH_Y_MIN = -1.0f;
const float SideWalkDetector::PASSTHROUGH_Y_MAX = -0.3f;
const double SideWalkDetector::RANSAC_DISTANCE_THRESHOLD = 0.05;
