#pragma once

#include <array>  // Defines the explicit model-channel to canonical-joint mapping.

#include "pose/PoseTypes.hpp"  // Provides the model-independent joint identifiers.

namespace video_engine {

inline constexpr std::array<JointId, 18> kOpenPoseCocoChannelToJoint = {
    JointId::Nose,
    JointId::Neck,
    JointId::RightShoulder,
    JointId::RightElbow,
    JointId::RightWrist,
    JointId::LeftShoulder,
    JointId::LeftElbow,
    JointId::LeftWrist,
    JointId::RightHip,
    JointId::RightKnee,
    JointId::RightAnkle,
    JointId::LeftHip,
    JointId::LeftKnee,
    JointId::LeftAnkle,
    JointId::RightEye,
    JointId::LeftEye,
    JointId::RightEar,
    JointId::LeftEar,
};

static_assert(kOpenPoseCocoChannelToJoint.size() == kJointCount,
              "OpenPose COCO must map all canonical joints.");

}  // namespace video_engine
