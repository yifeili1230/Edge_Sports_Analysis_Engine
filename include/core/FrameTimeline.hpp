#pragma once

#include <cmath>     // Validates finite source timestamps.
#include <cstdint>   // Stores monotonic frame identifiers.
#include <stdexcept> // Rejects invalid or non-monotonic source timestamps.

#include "core/FrameContext.hpp"  // Resets per-frame pose measurement state.

namespace video_engine {

class FrameTimeline {
public:
    void beginFrame(FrameContext& ctx, double source_time_seconds) {
        if (!std::isfinite(source_time_seconds) || source_time_seconds < 0.0) {
            throw std::runtime_error("Frame source timestamp must be finite and non-negative.");
        }
        if (next_frame_id_ > 1 && source_time_seconds <= last_source_time_seconds_) {
            throw std::runtime_error("Frame source timestamps must be strictly increasing.");
        }

        ctx.frame_id = next_frame_id_++;
        ctx.source_time_seconds = source_time_seconds;
        ctx.pose_inference_ran = false;
        ctx.pose_measurement_valid = false;
        ctx.poses.clear();
        last_source_time_seconds_ = source_time_seconds;
    }

private:
    std::uint64_t next_frame_id_ = 1;
    double last_source_time_seconds_ = 0.0;
};

}  // namespace video_engine
