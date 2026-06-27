#include <array>
#include <iostream>

#include "pose/OpenPoseCocoAdapter.hpp"
#include "pose/PoseTypes.hpp"

int main() {
    if (video_engine::kOpenPoseCocoChannelToJoint.size() != video_engine::kJointCount) {
        std::cerr << "OpenPose channel map does not cover the canonical schema" << std::endl;
        return 1;
    }

    std::array<bool, video_engine::kJointCount> mapped{};
    for (const auto joint : video_engine::kOpenPoseCocoChannelToJoint) {
        const auto index = video_engine::jointIndex(joint);
        if (mapped[index]) {
            std::cerr << "OpenPose channel map contains a duplicate joint" << std::endl;
            return 1;
        }
        mapped[index] = true;
    }

    video_engine::Pose pose;
    pose.person_id = 7;
    pose.frame_id = 42;
    pose.source_time_seconds = 1.4;
    auto& shoulder = pose.joint(video_engine::JointId::RightShoulder);
    shoulder.position_2d_pixels = video_engine::Point2D{212.5F, 104.25F};
    shoulder.confidence = 0.93F;
    shoulder.valid = true;

    if (pose.joint(video_engine::JointId::RightShoulder).position_2d_pixels.x != 212.5F ||
        shoulder.depth_meters.has_value() || shoulder.position_3d_meters.has_value()) {
        std::cerr << "Canonical 2D pose requires fake depth or lost float precision" << std::endl;
        return 1;
    }

    shoulder.depth_meters = 2.25F;
    shoulder.position_3d_meters = video_engine::Point3D{0.1F, 0.2F, 2.25F};
    if (!shoulder.depth_meters.has_value() || !shoulder.position_3d_meters.has_value()) {
        std::cerr << "Canonical pose could not accept optional depth data" << std::endl;
        return 1;
    }

    return 0;
}
