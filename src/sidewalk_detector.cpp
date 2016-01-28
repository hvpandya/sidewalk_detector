#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class SideWalkDetector
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber rgb_sub_, depth_sub;
  image_transport::Publisher rgb_pub_;

public:
  SideWalkDetector()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    rgb_sub_ = it_.subscribe("/camera/color/image_raw", 1, &SideWalkDetector::imageCb, this);
    rgb_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow("Output");
  }

  ~SideWalkDetector()
  {
    cv::destroyWindow("Output");
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Update GUI Window
    cv::imshow("Output", cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    rgb_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sidewalk_detector");
  SideWalkDetector sd;
  ros::spin();
  return 0;
}
