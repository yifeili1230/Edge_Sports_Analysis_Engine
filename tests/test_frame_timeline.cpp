#include <iostream>
#include <stdexcept>

#include "core/FrameTimeline.hpp"

int main() {
    video_engine::FrameTimeline timeline;
    video_engine::FrameContext ctx;

    timeline.beginFrame(ctx, 0.0);
    if (ctx.frame_id != 1 || ctx.pose_inference_ran || ctx.pose_measurement_valid ||
        !ctx.poses.empty()) {
        std::cerr << "First frame did not begin with a fresh invalid pose measurement" << std::endl;
        return 1;
    }

    ctx.pose_inference_ran = true;
    ctx.pose_measurement_valid = true;
    ctx.poses.push_back(video_engine::Pose{});

    timeline.beginFrame(ctx, 1.0 / 30.0);
    if (ctx.frame_id != 2 || ctx.pose_inference_ran || ctx.pose_measurement_valid ||
        !ctx.poses.empty()) {
        std::cerr << "Second frame reused pose state from the previous frame" << std::endl;
        return 1;
    }

    bool rejected_non_monotonic_time = false;
    try {
        timeline.beginFrame(ctx, 1.0 / 30.0);
    } catch (const std::runtime_error&) {
        rejected_non_monotonic_time = true;
    }
    if (!rejected_non_monotonic_time) {
        std::cerr << "Timeline accepted a duplicate source timestamp" << std::endl;
        return 1;
    }

    return 0;
}
