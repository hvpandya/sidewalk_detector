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
class SideWalkDetector
{
private:  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  typedef image_transport::SubscriberFilter ImageSubscriber;  

  ros::NodeHandle nh_;  
  image_transport::ImageTransport it_;
  image_transport::Publisher rgb_pub_;
  ros::Subscriber point_sub_;
  ros::Publisher in_point_pub_, out_point_pub_, filter_point_pub_;
  ImageSubscriber rgb_sub_;
  ImageSubscriber depth_sub_;   
  message_filters::Synchronizer<MySyncPolicy> sync_;
  

public:
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

  //callback function for rgb and depth
  void imageCb(const sensor_msgs::ImageConstPtr& rgb, const sensor_msgs::ImageConstPtr& depth)
  {
    std::stringstream filename;
    cv_bridge::CvImagePtr cv_ptr;    
    cv_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
    cv_bridge::CvImagePtr depth_ptr;    
    depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
    
    static int count=0, hue_avg=0, sat_avg=0;
    count++;
    int row = cv_ptr->image.size().height;
    int col = cv_ptr->image.size().width;

    cv::Size size(640,480);    
    cv::Mat dst;
    double min, max;

    //flip image
    cv::flip(cv_ptr->image, cv_ptr->image, -1);    
    
    //depth image processing
    cv::flip(depth_ptr->image, depth_ptr->image, -1);
    cv::resize(depth_ptr->image, depth_ptr->image, size);
    cv::Mat depth_src = depth_ptr->image.clone();
    cv::minMaxIdx(depth_src, &min, &max);
    depth_src.convertTo(depth_src,CV_8UC1, 255 / (max-min), -min*255 / (max-min));
     
    
    //size check
    //std::cout<<std::endl<<"RGB.W: "<<cv_ptr->image.size().width<<" RGB.H: "<<cv_ptr->image.size().height;
    
    //operation
    //cv::rectangle(cv_ptr->image, cv::Point(170,350), cv::Point(470,450), cv::Scalar(0,0,255), 1, 8, 0);
    cv::Mat op = cv_ptr->image.clone();
    cv::Mat src = cv_ptr->image.clone();
    cv::Mat roi = cv_ptr->image(cv::Rect(180,260,280,210));
    //cv::cvtColor(roi, roi, CV_BGR2HSV);
    

    //K-means Clustering
    cv::Mat samples(src.rows * src.cols, 3, CV_32F);
    for( int y = 0; y < src.rows; y++ )
      for( int x = 0; x < src.cols; x++ )
        for( int z = 0; z < 3; z++)
          samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y,x)[z];

    int clusterCount = 32;
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    kmeans(samples, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers );

    cv::Mat new_image(src.size(), src.type());
    for( int y = 0; y < src.rows; y++ )
      for( int x = 0; x < src.cols; x++ )
      { 
        int cluster_idx = labels.at<int>(y + x*src.rows,0);
        new_image.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
        new_image.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
        new_image.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
      }
    //imshow( "clustered image", new_image );

    //mean and stdev
    cv::Scalar mean, stdev;
    cv::meanStdDev(roi, mean, stdev);
    cv::Mat mask(src.size(), src.type(), cv::Scalar(0,0,0));    
    
    //Check each pixel and remove outliers
    for(int i=220;i<depth_src.size().height;i++)
    {
      for(int j=0;j<depth_src.size().width;j++)
      {
        cv::Vec3b val= src.at<cv::Vec3b>(i,j);
        if(val[0]>(mean[0]-2*stdev[0]) && val[0]<(mean[0]+2*stdev[0]) && val[1]>(mean[1]-2*stdev[1]) && val[1]<(mean[1]+2*stdev[1]) && val[2]>(mean[2]-3*stdev[2]) && val[2]<(mean[2]+3*stdev[2]) && (int)depth_src.at<uchar>(i,j)<40)     
        {
          //op.at<cv::Vec3b>(i,j)[0]=0;
          //op.at<cv::Vec3b>(i,j)[1]=0;
          op.at<cv::Vec3b>(i,j)[2]=255;
          //mask.at<cv::Vec3b>(i,j)[0]=val[0];
          //mask.at<cv::Vec3b>(i,j)[1]=val[1];
          mask.at<cv::Vec3b>(i,j)[2]=255;
        }
      }
    }

    //Publish result RGB image
    sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", op).toImageMsg();
    rgb_pub_.publish(rgb_msg);
    
    //display & save result
    //filename<<"img"<<count<<".jpg";
    //cv::imwrite(filename.str(),op);
    //cv::imshow("roi",roi);        
    //cv::imshow("RGB",op);
    //cv::applyColorMap(depth_src, depth_src, cv::COLORMAP_HOT);
    //cv::imshow("depth_src",depth_src);
    //cv::imshow("mask",mask);
    //cv::waitKey(3);
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sidewalk_detector");
  SideWalkDetector sd;

  while (ros::ok())
  ros::spin();

  return 0;
}
