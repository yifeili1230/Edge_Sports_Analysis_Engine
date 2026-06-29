#include <cmath>
#include <iostream>

#include <opencv2/core.hpp>

#include "pose/YoloPoseDecoder.hpp"

namespace {

void setFeature(cv::Mat& output, int feature, int proposal, float value) {
    const int proposals = output.size[2];
    output.ptr<float>()[feature * proposals + proposal] = value;
}

}  // namespace

int main() {
    const int dimensions[] = {1, 56, 2};
    cv::Mat output(3, dimensions, CV_32F, cv::Scalar(0));

    setFeature(output, 0, 0, 300.0F);
    setFeature(output, 1, 0, 300.0F);
    setFeature(output, 2, 0, 200.0F);
    setFeature(output, 3, 0, 300.0F);
    setFeature(output, 4, 0, 0.9F);

    for (int keypoint = 0; keypoint < 17; ++keypoint) {
        const int offset = 5 + keypoint * 3;
        setFeature(output, offset, 0, 100.0F + keypoint * 10.0F);
        setFeature(output, offset + 1, 0, 280.0F);
        setFeature(output, offset + 2, 0, 0.8F);
    }
    setFeature(output, 4, 1, 0.1F);

    video_engine::YoloLetterboxTransform transform;
    transform.scale = 1.0F;
    transform.pad_y = 80.0F;
    transform.original_width = 640;
    transform.original_height = 480;

    const auto poses = video_engine::YoloPoseDecoder::decode(
        output, transform, 7, 1.25, 0.25F, 0.25F, 0.45F, 5);
    if (poses.size() != 1 || !poses.front().valid ||
        poses.front().frame_id != 7 ||
        std::abs(poses.front().source_time_seconds - 1.25) > 0.001) {
        std::cerr << "YOLO pose decoder did not retain the valid person" << std::endl;
        return 1;
    }

    const auto& left_shoulder =
        poses.front().joint(video_engine::JointId::LeftShoulder);
    const auto& right_shoulder =
        poses.front().joint(video_engine::JointId::RightShoulder);
    const auto& neck = poses.front().joint(video_engine::JointId::Neck);
    if (!left_shoulder.valid || !right_shoulder.valid || !neck.valid ||
        std::abs(neck.position_2d_pixels.x -
                 (left_shoulder.position_2d_pixels.x +
                  right_shoulder.position_2d_pixels.x) *
                     0.5F) >
            0.001F ||
        std::abs(neck.position_2d_pixels.y - 200.0F) > 0.001F) {
        std::cerr << "COCO-17 mapping or synthesized neck is incorrect" << std::endl;
        return 1;
    }

    if (!poses.front().bounding_box.valid ||
        std::abs(poses.front().bounding_box.y - 70.0F) > 0.001F) {
        std::cerr << "YOLO letterbox coordinates were not restored" << std::endl;
        return 1;
    }
    return 0;
}
