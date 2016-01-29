//Author: Harsh Pandya
//Project: Sidewalk detector using RGB-D data

/*
Camera frames from topic header: camera_depth_optical_frame
                                 camera_color_optical_frame
Using static_tf_publisher with "map" fixed frame 
*/


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

//SideWalkDetector class with callback members for depth, RGB, and pcl
class SideWalkDetector
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber rgb_sub_, depth_sub_;
  image_transport::Publisher rgb_pub_, depth_pub_;
  ros::Subscriber point_sub_;
  ros::Publisher in_point_pub_, out_point_pub_, filter_point_pub_;

public:
  SideWalkDetector()
    : it_(nh_)
  {
    // Subscribers
    rgb_sub_ = it_.subscribe("/camera/color/image_raw", 1, &SideWalkDetector::rgbCb, this);
    depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &SideWalkDetector::depthCb, this);
    point_sub_= nh_.subscribe("/camera/depth/points", 1, &SideWalkDetector::pointCb, this);
    
    // Publishers
    rgb_pub_ = it_.advertise("/sidewalk_detector/color/image_raw", 1);    
    depth_pub_ = it_.advertise("/sidewalk_detector/depth/image_raw", 1);    
    in_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_in", 1);
    out_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_out", 1);
    filter_point_pub_= nh_.advertise<sensor_msgs::PointCloud2>("/sidewalk_detector/depth/points_filtered", 1);
    

  }
  
  /* Callback function for RGB data */
  void rgbCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;    
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    
    //flip image
    cv::flip(cv_ptr->image, cv_ptr->image, -1);
    
    //size check
    //std::cout<<std::endl<<"RGB.W: "<<cv_ptr->image.size().width<<" RGB.H: "<<cv_ptr->image.size().height;
    
    //GUI Window
    cv::namedWindow("RGB");
    cv::imshow("RGB",cv_ptr->image);
    cv::waitKey(3);

    //Output RGB stream
    rgb_pub_.publish(cv_ptr->toImageMsg());
  }

  /* Callback function for depth data */
  void depthCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr depth_ptr;    
    depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Size size(640, 480);    
    cv::Mat dst;
    double min, max;

    //flip image
    cv::flip(depth_ptr->image, depth_ptr->image, -1);

    //depth image data range adjustment for display
    cv::minMaxIdx(depth_ptr->image, &min, &max);
    depth_ptr->image.convertTo(dst,CV_8UC1, 255 / (max-min), -min);
  
    cv::resize(dst, dst, size);  
    //size check
    //std::cout<<std::endl<<"depth.W: "<<depth_ptr->image.size().width<<" depth.H: "<<depth_ptr->image.size().height;
    
    //GUI Window
    cv::namedWindow("depth");
    cv::imshow("depth", dst);
    cv::waitKey(3);

    //Output depth stream
    depth_pub_.publish(depth_ptr->toImageMsg());
  }


  /* Callback function for pcl data */
  void pointCb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    sensor_msgs::PointCloud2::Ptr point_in (new sensor_msgs::PointCloud2);
    sensor_msgs::PointCloud2::Ptr point_out (new sensor_msgs::PointCloud2);
    sensor_msgs::PointCloud2::Ptr point_filter (new sensor_msgs::PointCloud2);
    pcl::fromROSMsg (*msg, *cloud);

    //filter out points much above ground
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-1.0, -0.3);
    pass.filter (*cloud_filtered);
    pcl::toROSMsg (*cloud_filtered, *point_filter);
    pass.setFilterLimitsNegative (true);
    pass.filter (*cloud);

    //RANSAC plane segmentation
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.05);

    seg.setInputCloud(cloud_filtered);
    seg.segment (*inliers, *coefficients);

    //Extract inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_in);

    //Get the number of points associated with the planar surface
    //std::cout<<"Number of points in plane: "<<cloud_in->points.size ()<< std::endl;

    //Extract outliers from the output cloud
    extract.setNegative (true);
    extract.filter(*cloud_out);
    *cloud_out+=*cloud;
    pcl::toROSMsg (*cloud_in, *point_in);
    pcl::toROSMsg (*cloud_out, *point_out);
  
    //Publish output point cloud streams
    //filter_point_pub_.publish(point_filter);
    in_point_pub_.publish(point_in);
    out_point_pub_.publish(point_out);
 
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sidewalk_detector");
  SideWalkDetector sd;
  ros::spin();
  return 0;
}
