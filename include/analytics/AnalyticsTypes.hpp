#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "analytics/FilmSide.hpp"

namespace video_engine {

enum class SquatPhase {
    Unknown,
    Standing,
    Descending,
    Bottom,
    Ascending,
};

inline std::string squatPhaseName(SquatPhase phase) {
    switch (phase) {
        case SquatPhase::Standing:
            return "standing";
        case SquatPhase::Descending:
            return "descending";
        case SquatPhase::Bottom:
            return "bottom";
        case SquatPhase::Ascending:
            return "ascending";
        case SquatPhase::Unknown:
        default:
            return "unknown";
    }
}

enum class OhpPhase {
    Unknown,
    Rack,
    Pressing,
    Lockout,
    Lowering,
};

inline std::string ohpPhaseName(OhpPhase phase) {
    switch (phase) {
        case OhpPhase::Rack:
            return "rack";
        case OhpPhase::Pressing:
            return "pressing";
        case OhpPhase::Lockout:
            return "lockout";
        case OhpPhase::Lowering:
            return "lowering";
        case OhpPhase::Unknown:
        default:
            return "unknown";
    }
}

struct SquatRepSummary {
    int rep_index = 0;
    double start_time_seconds = 0.0;
    double end_time_seconds = 0.0;
    double descent_time_seconds = 0.0;
    double ascent_time_seconds = 0.0;
    double minimum_knee_angle_degrees = 0.0;
    double average_normalized_speed_per_second = 0.0;
    double peak_normalized_speed_per_second = 0.0;
};

struct OhpRepSummary {
    int rep_index = 0;
    double start_time_seconds = 0.0;
    double end_time_seconds = 0.0;
    double press_time_seconds = 0.0;
    double lowering_time_seconds = 0.0;
    double maximum_elbow_angle_degrees = 0.0;
    double wrist_travel_body_lengths = 0.0;
    double average_normalized_speed_per_second = 0.0;
    double peak_normalized_speed_per_second = 0.0;
};

struct PoseAnalysisResult {
    std::uint64_t frame_id = 0;
    double source_time_seconds = 0.0;
    bool valid = false;
    std::string exercise = "squat";
    SquatPhase phase = SquatPhase::Unknown;
    int completed_reps = 0;
    double knee_angle_degrees = 0.0;
    double hip_angle_degrees = 0.0;
    bool hip_angle_valid = false;
    double back_angle_degrees = 0.0;
    bool back_angle_valid = false;
    double ankle_angle_degrees = 0.0;
    bool ankle_angle_valid = false;
    double normalized_vertical_speed_per_second = 0.0;
    std::optional<FilmSide> film_side;
    std::optional<SquatRepSummary> completed_rep;
    OhpPhase ohp_phase = OhpPhase::Unknown;
    double elbow_angle_degrees = 0.0;
    double wrist_height_body_lengths = 0.0;
    std::optional<OhpRepSummary> completed_ohp_rep;
};

struct SquatSessionSummary {
    std::string source;
    std::size_t processed_frames = 0;
    std::size_t valid_analysis_frames = 0;
    std::size_t invalid_analysis_frames = 0;
    std::vector<SquatRepSummary> reps;
};

struct OhpSessionSummary {
    std::string source;
    std::size_t processed_frames = 0;
    std::size_t valid_analysis_frames = 0;
    std::size_t invalid_analysis_frames = 0;
    std::vector<OhpRepSummary> reps;
};

}  // namespace video_engine
