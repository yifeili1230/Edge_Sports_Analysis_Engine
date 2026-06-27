#pragma once

#include <array>    // Stores one observation for every canonical joint.
#include <cstddef>  // Provides std::size_t for joint indexing.
#include <cstdint>  // Stores stable frame and person identifiers.
#include <optional> // Represents depth and 3D data that monocular sources do not provide.

namespace video_engine {

enum class JointId : std::uint8_t {
    Nose,
    Neck,
    RightShoulder,
    RightElbow,
    RightWrist,
    LeftShoulder,
    LeftElbow,
    LeftWrist,
    RightHip,
    RightKnee,
    RightAnkle,
    LeftHip,
    LeftKnee,
    LeftAnkle,
    RightEye,
    LeftEye,
    RightEar,
    LeftEar,
    Count,
};

inline constexpr std::size_t kJointCount = static_cast<std::size_t>(JointId::Count);

constexpr std::size_t jointIndex(JointId joint) {
    return static_cast<std::size_t>(joint);
}

struct Point2D {
    float x = 0.0F;
    float y = 0.0F;
};

struct Point3D {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct FrameSize2D {
    float width = 0.0F;
    float height = 0.0F;
};

struct BoundingBox2D {
    float x = 0.0F;
    float y = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
    bool valid = false;
};

struct PoseKeypoint {
    Point2D position_2d_pixels;
    float confidence = 0.0F;
    bool valid = false;
    std::optional<float> depth_meters;
    std::optional<Point3D> position_3d_meters;
};

struct Pose {
    std::int64_t person_id = -1;
    std::uint64_t frame_id = 0;
    double source_time_seconds = 0.0;
    FrameSize2D frame_size_pixels;
    BoundingBox2D bounding_box;
    std::array<PoseKeypoint, kJointCount> joints{};
    bool valid = false;

    PoseKeypoint& joint(JointId id) {
        return joints[jointIndex(id)];
    }

    const PoseKeypoint& joint(JointId id) const {
        return joints[jointIndex(id)];
    }
};

}  // namespace video_engine
