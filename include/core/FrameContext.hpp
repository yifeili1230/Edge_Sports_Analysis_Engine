#pragma once

#include <cstdint> // Stores monotonic frame identifiers.
#include <string>  // Stores metadata and profiling stage names.
#include <vector>  // Stores poses and timing arrays.
#include <optional> // Stores an optional exercise-analysis result for the current frame.

#include <opencv2/core.hpp>  // Provides cv::Mat.

#include "pose/PoseTypes.hpp"  // Provides model-independent pose measurements.
#include "analytics/AnalyticsTypes.hpp" // Provides live analytics and completed-rep events.

namespace video_engine {

struct FrameContext {
    cv::Mat raw_frame;
    cv::Mat processed_frame;
    std::uint64_t frame_id = 0;
    double source_time_seconds = 0.0;
    bool pose_inference_ran = false;
    bool pose_measurement_valid = false;
    std::vector<Pose> poses;
    std::optional<PoseAnalysisResult> pose_analysis;
    std::vector<SquatRepSummary> completed_rep_events;
    std::vector<OhpRepSummary> completed_ohp_rep_events;
    std::vector<std::string> stage_names;
    std::vector<double> stage_latencies_ms;
    std::string metadata;
};

}  // namespace video_engine
