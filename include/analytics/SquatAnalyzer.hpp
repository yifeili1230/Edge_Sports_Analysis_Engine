#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "analytics/IPoseAnalyzer.hpp"

namespace video_engine {

struct SquatAnalyzerConfig {
    double minimum_joint_confidence = 0.12;
    FilmSide film_side = FilmSide::RightSideView;
    double standing_angle_degrees = 160.0;
    double descent_start_angle_degrees = 155.0;
    double descent_commit_angle_degrees = 145.0;
    double bottom_angle_degrees = 100.0;
    double bottom_exit_angle_degrees = 105.0;
    double minimum_normalized_speed_per_second = 0.02;
    double direction_confirmation_speed_per_second = 0.50;
    double smoothing_alpha = 0.35;
    int transition_confirmation_frames = 2;
    int bottom_confirmation_frames = 1;
    double minimum_phase_duration_seconds = 0.50;
    double minimum_rep_duration_seconds = 0.50;
    double maximum_knee_angular_velocity_degrees_per_second = 720.0;
    double maximum_normalized_speed_per_second = 4.0;
    double minimum_shoulder_travel_body_lengths = 0.15;
};

class SquatAnalyzer : public IPoseAnalyzer {
public:
    explicit SquatAnalyzer(SquatAnalyzerConfig config = {}) : config_(config) {
        if (config_.smoothing_alpha <= 0.0 || config_.smoothing_alpha > 1.0) {
            throw std::invalid_argument("Squat smoothing alpha must be in (0, 1].");
        }
        if (config_.minimum_normalized_speed_per_second <= 0.0) {
            throw std::invalid_argument("Squat minimum speed must be positive.");
        }
        if (config_.minimum_shoulder_travel_body_lengths <= 0.0) {
            throw std::invalid_argument(
                "Squat minimum shoulder travel must be positive.");
        }
    }

    PoseAnalysisResult analyze(const Pose* pose, std::uint64_t frame_id,
                               double source_time_seconds) override {
        PoseAnalysisResult result;
        result.frame_id = frame_id;
        result.source_time_seconds = source_time_seconds;
        result.phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());
        if (!completed_reps_.empty()) {
            result.completed_rep = completed_reps_.back();
        }

        const SideObservation side = selectSide(pose);
        if (!side.valid) {
            ++invalid_frames_;
            previous_shoulder_valid_ = false;
            return result;
        }

        ++valid_frames_;
        result.valid = true;
        result.film_side = config_.film_side;
        result.hip_angle_degrees =
            angleDegrees(side.shoulder, side.hip, side.knee);
        result.knee_angle_degrees =
            angleDegrees(side.hip, side.knee, side.ankle);
        result.back_angle_degrees = angleFromVertical(side.shoulder, side.hip);
        result.ankle_angle_degrees = angleFromFloor(side.knee, side.ankle);
        result.hip_angle_valid = true;
        result.back_angle_valid = true;
        result.ankle_angle_valid = true;

        if (!smoothed_shoulder_valid_) {
            smoothed_shoulder_y_ = side.shoulder.y;
            smoothed_shoulder_valid_ = true;
        } else {
            smoothed_shoulder_y_ =
                smooth(smoothed_shoulder_y_, side.shoulder.y, config_.smoothing_alpha);
        }

        double speed = 0.0;
        if (previous_shoulder_valid_ &&
            source_time_seconds > previous_source_time_seconds_) {
            speed = (smoothed_shoulder_y_ - previous_shoulder_y_) /
                    side.body_scale /
                    (source_time_seconds - previous_source_time_seconds_);
        }
        result.normalized_vertical_speed_per_second = speed;

        updateState(source_time_seconds, smoothed_shoulder_y_, side.body_scale,
                    result.knee_angle_degrees, speed, result);

        result.phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());
        if (!completed_reps_.empty()) {
            result.completed_rep = completed_reps_.back();
        }

        previous_shoulder_y_ = smoothed_shoulder_y_;
        previous_source_time_seconds_ = source_time_seconds;
        previous_shoulder_valid_ = true;
        return result;
    }

    const std::vector<SquatRepSummary>& completedReps() const override {
        return completed_reps_;
    }

    FilmSide filmSide() const noexcept override {
        return config_.film_side;
    }

    std::size_t validFrameCount() const override {
        return valid_frames_;
    }

    std::size_t invalidFrameCount() const override {
        return invalid_frames_;
    }

    static double angleDegrees(const Point2D& first, const Point2D& vertex,
                               const Point2D& third) {
        const double first_x = static_cast<double>(first.x) - vertex.x;
        const double first_y = static_cast<double>(first.y) - vertex.y;
        const double third_x = static_cast<double>(third.x) - vertex.x;
        const double third_y = static_cast<double>(third.y) - vertex.y;
        const double first_length = std::hypot(first_x, first_y);
        const double third_length = std::hypot(third_x, third_y);
        if (first_length <= std::numeric_limits<double>::epsilon() ||
            third_length <= std::numeric_limits<double>::epsilon()) {
            return 0.0;
        }
        const double cosine = std::clamp(
            (first_x * third_x + first_y * third_y) /
                (first_length * third_length),
            -1.0, 1.0);
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
        return std::acos(cosine) * kRadiansToDegrees;
    }

private:
    struct SideObservation {
        bool valid = false;
        Point2D shoulder;
        Point2D hip;
        Point2D knee;
        Point2D ankle;
        double confidence = 0.0;
        double body_scale = 1.0;
    };

    SideObservation makeSide(const Pose* pose, JointId shoulder_id, JointId hip_id,
                             JointId knee_id, JointId ankle_id) const {
        SideObservation side;
        if (pose == nullptr || !pose->valid) {
            return side;
        }

        const auto& shoulder = pose->joint(shoulder_id);
        const auto& hip = pose->joint(hip_id);
        const auto& knee = pose->joint(knee_id);
        const auto& ankle = pose->joint(ankle_id);
        const auto usable = [this](const PoseKeypoint& point) {
            return point.valid &&
                   point.confidence >= config_.minimum_joint_confidence;
        };
        if (!usable(shoulder) || !usable(hip) || !usable(knee) ||
            !usable(ankle)) {
            return side;
        }

        side.shoulder = shoulder.position_2d_pixels;
        side.hip = hip.position_2d_pixels;
        side.knee = knee.position_2d_pixels;
        side.ankle = ankle.position_2d_pixels;
        side.confidence =
            std::min({static_cast<double>(shoulder.confidence),
                      static_cast<double>(hip.confidence),
                      static_cast<double>(knee.confidence),
                      static_cast<double>(ankle.confidence)});
        side.body_scale =
            std::hypot(static_cast<double>(side.shoulder.x - side.hip.x),
                       static_cast<double>(side.shoulder.y - side.hip.y));
        side.valid = side.body_scale > 1.0;
        return side;
    }

    SideObservation selectSide(const Pose* pose) const {
        if (filmSidePrefersLeftBodyChain(config_.film_side)) {
            return makeSide(pose, JointId::LeftShoulder, JointId::LeftHip,
                            JointId::LeftKnee, JointId::LeftAnkle);
        }
        if (filmSidePrefersRightBodyChain(config_.film_side)) {
            return makeSide(pose, JointId::RightShoulder, JointId::RightHip,
                            JointId::RightKnee, JointId::RightAnkle);
        }

        const SideObservation left =
            makeSide(pose, JointId::LeftShoulder, JointId::LeftHip,
                     JointId::LeftKnee, JointId::LeftAnkle);
        const SideObservation right =
            makeSide(pose, JointId::RightShoulder, JointId::RightHip,
                     JointId::RightKnee, JointId::RightAnkle);
        if (!left.valid) {
            return right;
        }
        if (!right.valid) {
            return left;
        }
        return left.confidence >= right.confidence ? left : right;
    }

    static double smooth(double previous, double current, double alpha) {
        return alpha * current + (1.0 - alpha) * previous;
    }

    static double angleFromVertical(const Point2D& top, const Point2D& bottom) {
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
        return std::atan2(std::abs(static_cast<double>(top.x - bottom.x)),
                          std::abs(static_cast<double>(top.y - bottom.y))) *
               kRadiansToDegrees;
    }

    // The pose schema has no foot keypoint, so this is the shin angle to the floor.
    static double angleFromFloor(const Point2D& knee, const Point2D& ankle) {
        constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
        return std::atan2(std::abs(static_cast<double>(knee.y - ankle.y)),
                          std::abs(static_cast<double>(knee.x - ankle.x))) *
               kRadiansToDegrees;
    }

    void beginRep(double source_time_seconds, double shoulder_y,
                  double body_scale, double knee_angle) {
        phase_ = SquatPhase::Descending;
        rep_start_time_seconds_ = source_time_seconds;
        rep_start_shoulder_y_ = shoulder_y;
        deepest_shoulder_y_ = shoulder_y;
        rep_body_scale_ = body_scale;
        minimum_knee_angle_ = knee_angle;
        speed_sum_ = 0.0;
        speed_samples_ = 0;
        peak_speed_ = 0.0;
    }

    void recordRepFrame(double knee_angle, double speed) {
        minimum_knee_angle_ = std::min(minimum_knee_angle_, knee_angle);
        const double magnitude = std::abs(speed);
        speed_sum_ += magnitude;
        ++speed_samples_;
        peak_speed_ = std::max(peak_speed_, magnitude);
    }

    void finishRep(double source_time_seconds, PoseAnalysisResult& result) {
        SquatRepSummary rep;
        rep.rep_index = static_cast<int>(completed_reps_.size()) + 1;
        rep.start_time_seconds = rep_start_time_seconds_;
        rep.end_time_seconds = source_time_seconds;
        rep.descent_time_seconds =
            std::max(0.0, bottom_time_seconds_ - rep_start_time_seconds_);
        rep.ascent_time_seconds =
            std::max(0.0, source_time_seconds - bottom_time_seconds_);
        rep.minimum_knee_angle_degrees = minimum_knee_angle_;
        rep.average_normalized_speed_per_second =
            speed_samples_ == 0 ? 0.0 : speed_sum_ / speed_samples_;
        rep.peak_normalized_speed_per_second = peak_speed_;
        completed_reps_.push_back(rep);
        result.completed_rep = rep;
        phase_ = SquatPhase::Standing;
    }

    void updateState(double source_time_seconds, double shoulder_y,
                     double body_scale, double knee_angle, double speed,
                     PoseAnalysisResult& result) {
        const double threshold = config_.minimum_normalized_speed_per_second;
        const bool descending = speed > threshold;
        const bool ascending = speed < -threshold;

        if (phase_ == SquatPhase::Unknown) {
            phase_ = SquatPhase::Standing;
        }

        if (phase_ == SquatPhase::Standing && descending) {
            beginRep(source_time_seconds, previous_shoulder_y_, body_scale,
                     knee_angle);
            recordRepFrame(knee_angle, speed);
            return;
        }

        if (phase_ == SquatPhase::Descending) {
            recordRepFrame(knee_angle, speed);
            deepest_shoulder_y_ = std::max(deepest_shoulder_y_, shoulder_y);
            if (ascending) {
                const double travel =
                    (deepest_shoulder_y_ - rep_start_shoulder_y_) /
                    rep_body_scale_;
                if (travel >= config_.minimum_shoulder_travel_body_lengths) {
                    bottom_time_seconds_ = source_time_seconds;
                    phase_ = SquatPhase::Ascending;
                } else {
                    phase_ = SquatPhase::Standing;
                }
            }
            return;
        }

        if (phase_ == SquatPhase::Ascending) {
            recordRepFrame(knee_angle, speed);
            if (descending) {
                deepest_shoulder_y_ = std::max(deepest_shoulder_y_, shoulder_y);
                phase_ = SquatPhase::Descending;
                return;
            }
            const double return_tolerance = rep_body_scale_ * 0.05;
            const bool returned_to_start =
                shoulder_y <= rep_start_shoulder_y_ + return_tolerance;
            if (returned_to_start) {
                finishRep(source_time_seconds, result);
            }
        }
    }

    SquatAnalyzerConfig config_;
    SquatPhase phase_ = SquatPhase::Unknown;
    std::vector<SquatRepSummary> completed_reps_;
    std::size_t valid_frames_ = 0;
    std::size_t invalid_frames_ = 0;
    bool smoothed_shoulder_valid_ = false;
    bool previous_shoulder_valid_ = false;
    double smoothed_shoulder_y_ = 0.0;
    double previous_shoulder_y_ = 0.0;
    double previous_source_time_seconds_ = 0.0;
    double rep_start_time_seconds_ = 0.0;
    double rep_start_shoulder_y_ = 0.0;
    double deepest_shoulder_y_ = 0.0;
    double rep_body_scale_ = 1.0;
    double bottom_time_seconds_ = 0.0;
    double minimum_knee_angle_ = 180.0;
    double speed_sum_ = 0.0;
    std::size_t speed_samples_ = 0;
    double peak_speed_ = 0.0;
};

}  // namespace video_engine
