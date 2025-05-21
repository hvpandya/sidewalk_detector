#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

// Replicated constants from SideWalkDetector for testing purposes
// In a real application, these might be exposed via a header or config
namespace TestConstants {
    const float PASSTHROUGH_Y_MIN = -1.0f;
    const float PASSTHROUGH_Y_MAX = -0.3f;
    const double RANSAC_DISTANCE_THRESHOLD = 0.05;
} // namespace TestConstants

// Test fixture for PCL processing tests
class PCLProcessingTest : public ::testing::Test {
protected:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_;

    void SetUp() override {
        cloud_in_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_out_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        cloud_filtered_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    }
};

TEST_F(PCLProcessingTest, PassthroughFilterTest) {
    // Create a sample point cloud
    cloud_in_->points.emplace_back(0.0f, -1.5f, 0.0f); // Outside (below min)
    cloud_in_->points.emplace_back(0.0f, -0.5f, 0.0f); // Inside
    cloud_in_->points.emplace_back(0.0f,  0.0f, 0.0f); // Outside (above max)
    cloud_in_->points.emplace_back(0.0f, -0.8f, 1.0f); // Inside
    cloud_in_->points.emplace_back(0.0f, -0.2f, 1.0f); // Outside (above max)
    cloud_in_->width = cloud_in_->points.size();
    cloud_in_->height = 1;

    // Apply Passthrough filter logic
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud_in_);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(TestConstants::PASSTHROUGH_Y_MIN, TestConstants::PASSTHROUGH_Y_MAX);
    pass.filter(*cloud_filtered_);

    // Assertions
    ASSERT_EQ(cloud_filtered_->points.size(), 2);
    for (const auto& point : cloud_filtered_->points) {
        EXPECT_GE(point.y, TestConstants::PASSTHROUGH_Y_MIN);
        EXPECT_LE(point.y, TestConstants::PASSTHROUGH_Y_MAX);
    }

    // Test the negative filter part as well (points outside the limits)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outside_limits(new pcl::PointCloud<pcl::PointXYZ>);
    pass.setFilterLimitsNegative(true);
    pass.filter(*cloud_outside_limits);
    ASSERT_EQ(cloud_outside_limits->points.size(), 3);
     for (const auto& point : cloud_outside_limits->points) {
        EXPECT_TRUE(point.y < TestConstants::PASSTHROUGH_Y_MIN || point.y > TestConstants::PASSTHROUGH_Y_MAX);
    }
}

TEST_F(PCLProcessingTest, RansacPlaneSegmentationTest) {
    // Create a sample point cloud with a clear plane and outliers
    // Planar points (on z=0 plane)
    cloud_in_->points.emplace_back(0.0f, 0.0f, 0.0f);
    cloud_in_->points.emplace_back(1.0f, 0.0f, 0.01f); // Slightly off but within threshold
    cloud_in_->points.emplace_back(0.0f, 1.0f, -0.01f); // Slightly off but within threshold
    cloud_in_->points.emplace_back(1.0f, 1.0f, 0.0f);
    cloud_in_->points.emplace_back(0.5f, 0.5f, 0.02f); // Slightly off

    // Outlier points
    cloud_in_->points.emplace_back(0.0f, 0.0f, 1.0f);
    cloud_in_->points.emplace_back(1.0f, 0.0f, 1.5f);
    cloud_in_->points.emplace_back(0.0f, 1.0f, -1.0f);
    cloud_in_->width = cloud_in_->points.size();
    cloud_in_->height = 1;

    // Apply RANSAC plane segmentation logic
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(TestConstants::RANSAC_DISTANCE_THRESHOLD);
    seg.setInputCloud(cloud_in_);
    seg.segment(*inliers, *coefficients);

    ASSERT_TRUE(inliers->indices.size() > 0) << "RANSAC should find some inliers for the plane.";
    
    if (inliers->indices.size() > 0) {
        // Extract inliers
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_in_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_filtered_); // cloud_filtered_ now holds inliers

        // Extract outliers
        extract.setNegative(true);
        extract.filter(*cloud_out_); // cloud_out_ now holds outliers

        // Assertions (exact numbers can be tricky with RANSAC, but we expect most planar points)
        EXPECT_GE(cloud_filtered_->points.size(), 4); // Expect at least the 4 clear planar points + maybe the slightly off one
        EXPECT_LE(cloud_out_->points.size(), 4);      // Expect the 3 clear outliers + maybe one slightly off planar one if threshold is tight

        // Check if inliers are close to a plane (e.g., check their Z values if plane is XY)
        // For z=0 plane, coefficients->values[3] (d) should be close to 0
        // and coefficients->values[2] (c for normal vector) should be close to 1 or -1
        ASSERT_EQ(coefficients->values.size(), 4); // a, b, c, d
        // Example check for a plane close to z=0. This depends heavily on the generated plane.
        // EXPECT_NEAR(coefficients->values[2], 1.0, 0.1); // Normal vector's Z component
        // EXPECT_NEAR(coefficients->values[3], 0.0, TestConstants::RANSAC_DISTANCE_THRESHOLD * 2); // Plane equation's d component

        // Verify that known outliers are in the cloud_out_
        bool outlier1_found = false;
        bool outlier2_found = false;
        bool outlier3_found = false;
        for(const auto& p : cloud_out_->points) {
            if (std::abs(p.z - 1.0f) < 0.001f) outlier1_found = true;
            if (std::abs(p.z - 1.5f) < 0.001f) outlier2_found = true;
            if (std::abs(p.z - (-1.0f)) < 0.001f) outlier3_found = true;
        }
        EXPECT_TRUE(outlier1_found);
        EXPECT_TRUE(outlier2_found);
        EXPECT_TRUE(outlier3_found);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    // ros::init(argc, argv, "tester_pcl_processing"); // Optional: if ROS functionalities are directly needed
    return RUN_ALL_TESTS();
}
