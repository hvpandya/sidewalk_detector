#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> // For CV_32FC1, CV_8UC1 if not in imgproc
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h> // For sensor_msgs::image_encodings

// Replicated constants from SideWalkDetector for testing purposes
namespace TestConstants {
    // For imageCb and helpers
    const cv::Rect ROI_RECT(180, 260, 280, 210); // Example, adjust if needed for a specific test
    const uchar MAX_DEPTH_FOR_SIDEWALK = 40;
    const int CLASSIFICATION_START_ROW = 220; // Not directly used in pixel test, but part of logic
    const double STDEV_MULT_B = 2.0;
    const double STDEV_MULT_G = 2.0;
    const double STDEV_MULT_R = 3.0;
    const cv::Size TARGET_IMAGE_SIZE(640, 480);
} // namespace TestConstants


// Replicated logic from SideWalkDetector's process_depth_image
// In a real application, this would ideally be a free function or static public method
cv::Mat replicate_process_depth_image(const cv::Mat& raw_depth_image, const cv::Size& target_size) {
    cv::Mat depth_image_flipped;
    cv::flip(raw_depth_image, depth_image_flipped, -1); // Flip vertically

    cv::Mat depth_image_resized;
    cv::resize(depth_image_flipped, depth_image_resized, target_size);

    cv::Mat depth_src_normalized;
    double min_val, max_val;
    cv::minMaxIdx(depth_image_resized, &min_val, &max_val);

    if (max_val - min_val > std::numeric_limits<double>::epsilon()) {
        depth_image_resized.convertTo(depth_src_normalized, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    } else {
        depth_image_resized.convertTo(depth_src_normalized, CV_8UC1, 1.0, 0);
    }
    return depth_src_normalized;
}

// Replicated logic from SideWalkDetector's classify_sidewalk_pixels (for a single pixel)
// In a real application, this would ideally be a free function or static public method
void replicate_classify_single_pixel(cv::Vec3b& output_pixel_color, // Output color (e.g., from an op image)
                                     const cv::Vec3b& current_pixel_color, // Input color from src
                                     const uchar current_pixel_depth,     // Input depth from processed depth_src
                                     const cv::Scalar& roi_mean,
                                     const cv::Scalar& roi_stdev) {
    // Default: pixel is not sidewalk
    // output_pixel_color remains unchanged unless classified as sidewalk

    if (current_pixel_color[0] > (roi_mean[0] - TestConstants::STDEV_MULT_B * roi_stdev[0]) && current_pixel_color[0] < (roi_mean[0] + TestConstants::STDEV_MULT_B * roi_stdev[0]) &&
        current_pixel_color[1] > (roi_mean[1] - TestConstants::STDEV_MULT_G * roi_stdev[1]) && current_pixel_color[1] < (roi_mean[1] + TestConstants::STDEV_MULT_G * roi_stdev[1]) &&
        current_pixel_color[2] > (roi_mean[2] - TestConstants::STDEV_MULT_R * roi_stdev[2]) && current_pixel_color[2] < (roi_mean[2] + TestConstants::STDEV_MULT_R * roi_stdev[2]) &&
        current_pixel_depth < TestConstants::MAX_DEPTH_FOR_SIDEWALK) {
        // Mark detected sidewalk pixels in red
        output_pixel_color[0] = 0;   // Blue
        output_pixel_color[1] = 0;   // Green
        output_pixel_color[2] = 255; // Red
    }
}


TEST(DepthImageProcessingTest, HandlesNormalizationAndResize) {
    // Create a sample raw depth image (32FC1)
    // For simplicity, make it smaller and then resize.
    cv::Mat raw_depth = cv::Mat::zeros(240, 320, CV_32FC1);
    // Add some depth values
    raw_depth.at<float>(10, 10) = 1.0f; // Min depth after flip
    raw_depth.at<float>(20, 20) = 5.0f; // Max depth after flip
    raw_depth.at<float>(30, 30) = 2.5f;

    // Simulate the original image being upside down, so (10,10) which is near top-left
    // becomes near bottom-left after flip, and its value (1.0f) will be min.
    // (20,20) with 5.0f will be max.

    cv::Mat processed_depth = replicate_process_depth_image(raw_depth, TestConstants::TARGET_IMAGE_SIZE);

    ASSERT_EQ(processed_depth.rows, TestConstants::TARGET_IMAGE_SIZE.height);
    ASSERT_EQ(processed_depth.cols, TestConstants::TARGET_IMAGE_SIZE.width);
    ASSERT_EQ(processed_depth.type(), CV_8UC1);

    double min_val, max_val;
    cv::minMaxIdx(processed_depth, &min_val, &max_val);

    // After normalization, min depth should map to 0 and max to 255
    // (Allow for small floating point inaccuracies if any were introduced by resize)
    EXPECT_NEAR(min_val, 0, 1.0); 
    EXPECT_NEAR(max_val, 255, 1.0);

    // Check a specific normalized value (optional, more complex due to resize and flip)
    // Example: find the flipped and resized coordinate of raw_depth.at<float>(30,30) = 2.5f
    // Original (y=30, x=30) in 240x320. Flipped y_flipped = 240-1-30 = 209.
    // Resized: y_resized = 209 * (480/240) = 418. x_resized = 30 * (640/320) = 60.
    // Expected normalized value: (2.5 - 1.0) / (5.0 - 1.0) * 255 = 1.5 / 4.0 * 255 = 0.375 * 255 = 95.625
    // This is highly dependent on interpolation during resize. A simpler check is often better.
    // For instance, if we had a pixel with value 1.0f (min) and one with 5.0f (max),
    // we'd expect them to be 0 and 255 respectively in the normalized image,
    // provided they don't get lost/averaged out by the resize.
}

TEST(DepthImageProcessingTest, HandlesAllSameDepthValues) {
    cv::Mat raw_depth = cv::Mat(240, 320, CV_32FC1, cv::Scalar(2.0f)); // All pixels have depth 2.0

    cv::Mat processed_depth = replicate_process_depth_image(raw_depth, TestConstants::TARGET_IMAGE_SIZE);

    ASSERT_EQ(processed_depth.rows, TestConstants::TARGET_IMAGE_SIZE.height);
    ASSERT_EQ(processed_depth.cols, TestConstants::TARGET_IMAGE_SIZE.width);
    ASSERT_EQ(processed_depth.type(), CV_8UC1);

    double min_val, max_val;
    cv::minMaxIdx(processed_depth, &min_val, &max_val);
    
    // If all input values are the same, the normalization should result in 0 (or some other consistent value based on implementation)
    EXPECT_EQ(min_val, 0); 
    EXPECT_EQ(max_val, 0);
}


TEST(SidewalkPixelClassificationTest, PixelMeetsCriteria) {
    cv::Vec3b color_pixel(100, 110, 120); // BGR
    uchar depth_pixel_value = 20;         // Well below MAX_DEPTH_FOR_SIDEWALK

    cv::Scalar mean_roi_color(90, 100, 110); // BGR mean
    cv::Scalar stdev_roi_color(10, 10, 10);  // BGR stdev

    cv::Vec3b output_color = color_pixel; // Initialize output with original color
    replicate_classify_single_pixel(output_color, color_pixel, depth_pixel_value, mean_roi_color, stdev_roi_color);

    // Expect pixel to be marked red
    EXPECT_EQ(output_color[0], 0);   // Blue
    EXPECT_EQ(output_color[1], 0);   // Green
    EXPECT_EQ(output_color[2], 255); // Red
}

TEST(SidewalkPixelClassificationTest, PixelFailsColorCriteria) {
    cv::Vec3b color_pixel(50, 60, 70);   // Far from mean
    uchar depth_pixel_value = 20;        // Depth is fine

    cv::Scalar mean_roi_color(100, 110, 120);
    cv::Scalar stdev_roi_color(5, 5, 5); // Smaller stdev to make it fail easily

    cv::Vec3b initial_output_color = color_pixel;
    cv::Vec3b output_color = initial_output_color;
    replicate_classify_single_pixel(output_color, color_pixel, depth_pixel_value, mean_roi_color, stdev_roi_color);

    // Expect pixel NOT to be marked red (should remain original color)
    EXPECT_EQ(output_color, initial_output_color);
}

TEST(SidewalkPixelClassificationTest, PixelFailsDepthCriteria) {
    cv::Vec3b color_pixel(100, 110, 120); // Color is fine
    uchar depth_pixel_value = 50;         // Depth is too high (>= MAX_DEPTH_FOR_SIDEWALK)

    cv::Scalar mean_roi_color(90, 100, 110);
    cv::Scalar stdev_roi_color(10, 10, 10);

    cv::Vec3b initial_output_color = color_pixel;
    cv::Vec3b output_color = initial_output_color;
    replicate_classify_single_pixel(output_color, color_pixel, depth_pixel_value, mean_roi_color, stdev_roi_color);

    // Expect pixel NOT to be marked red
    EXPECT_EQ(output_color, initial_output_color);
}

TEST(SidewalkPixelClassificationTest, PixelFailsBothCriteria) {
    cv::Vec3b color_pixel(10, 20, 30);    // Color is off
    uchar depth_pixel_value = 100;       // Depth is off

    cv::Scalar mean_roi_color(90, 100, 110);
    cv::Scalar stdev_roi_color(10, 10, 10);

    cv::Vec3b initial_output_color = color_pixel;
    cv::Vec3b output_color = initial_output_color;
    replicate_classify_single_pixel(output_color, color_pixel, depth_pixel_value, mean_roi_color, stdev_roi_color);

    // Expect pixel NOT to be marked red
    EXPECT_EQ(output_color, initial_output_color);
}

// Note: No main() function here, as it's expected to be in test_pcl_processing.cpp
// and both files will be linked into a single test executable.
