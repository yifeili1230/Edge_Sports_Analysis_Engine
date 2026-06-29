#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "analytics/IPoseAnalyzer.hpp"

namespace video_engine {

struct OhpAnalyzerConfig {
    double minimum_joint_confidence = 0.12;
    FilmSide film_side = FilmSide::RearView;
    double rack_elbow_angle_degrees = 120.0;
    double lockout_elbow_angle_degrees = 155.0;
    double lockout_exit_angle_degrees = 145.0;
    double minimum_normalized_speed_per_second = 0.04;
    double smoothing_alpha = 0.35;
    double minimum_wrist_travel_body_lengths = 0.35;
    double rack_return_tolerance_body_lengths = 0.20;
    double minimum_rep_duration_seconds = 0.50;
};

class OhpAnalyzer : public IPoseAnalyzer {
public:
    explicit OhpAnalyzer(OhpAnalyzerConfig config = {}) : config_(config) {
        if (config_.smoothing_alpha <= 0.0 || config_.smoothing_alpha > 1.0) {
            throw std::invalid_argument("OHP smoothing alpha must be in (0, 1].");
        }
        if (config_.minimum_normalized_speed_per_second <= 0.0 ||
            config_.minimum_wrist_travel_body_lengths <= 0.0 ||
            config_.minimum_rep_duration_seconds < 0.0) {
            throw std::invalid_argument(
                "OHP speed, travel, and duration thresholds must be positive.");
        }
        if (config_.rack_elbow_angle_degrees >=
                config_.lockout_exit_angle_degrees ||
            config_.lockout_exit_angle_degrees >=
                config_.lockout_elbow_angle_degrees) {
            throw std::invalid_argument(
                "OHP elbow thresholds must satisfy rack < lockout exit < lockout.");
        }
    }

    PoseAnalysisResult analyze(const Pose* pose, std::uint64_t frame_id,
                               double source_time_seconds) override {
        PoseAnalysisResult result;
        result.frame_id = frame_id;
        result.source_time_seconds = source_time_seconds;
        result.exercise = "ohp";
        result.ohp_phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());
        if (!completed_reps_.empty()) {
            result.completed_ohp_rep = completed_reps_.back();
        }

        const ArmObservation observation = observe(pose);
        if (!observation.valid) {
            ++invalid_frames_;
            previous_wrist_valid_ = false;
            return result;
        }

        ++valid_frames_;
        result.valid = true;
        result.film_side = config_.film_side;
        result.elbow_angle_degrees = observation.elbow_angle_degrees;
        result.wrist_height_body_lengths =
            (observation.shoulder_y - observation.wrist_y) /
            observation.body_scale;

        if (!smoothed_wrist_valid_) {
            smoothed_wrist_y_ = observation.wrist_y;
            smoothed_wrist_valid_ = true;
        } else {
            smoothed_wrist_y_ =
                smooth(smoothed_wrist_y_, observation.wrist_y,
                       config_.smoothing_alpha);
        }

        double speed = 0.0;
        if (previous_wrist_valid_ &&
            source_time_seconds > previous_source_time_seconds_) {
            speed = (smoothed_wrist_y_ - previous_wrist_y_) /
                    observation.body_scale /
                    (source_time_seconds - previous_source_time_seconds_);
        }
        result.normalized_vertical_speed_per_second = speed;

        updateState(source_time_seconds, smoothed_wrist_y_,
                    observation.body_scale, observation.elbow_angle_degrees,
                    speed, result);

        result.ohp_phase = phase_;
        result.completed_reps = static_cast<int>(completed_reps_.size());
        if (!completed_reps_.empty()) {
            result.completed_ohp_rep = completed_reps_.back();
        }
        previous_wrist_y_ = smoothed_wrist_y_;
        previous_source_time_seconds_ = source_time_seconds;
        previous_wrist_valid_ = true;
        return result;
    }

    FilmSide filmSide() const noexcept override {
        return config_.film_side;
    }

    const std::vector<SquatRepSummary>& completedReps() const override {
        static const std::vector<SquatRepSummary> no_squat_reps;
        return no_squat_reps;
    }

    const std::vector<OhpRepSummary>& completedOhpReps() const override {
        return completed_reps_;
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
    struct SideArm {
        bool valid = false;
        Point2D shoulder;
        Point2D elbow;
        Point2D wrist;
        Point2D hip;
        double confidence = 0.0;
        double body_scale = 0.0;
        double elbow_angle_degrees = 0.0;
    };

    struct ArmObservation {
        bool valid = false;
        double shoulder_y = 0.0;
        double wrist_y = 0.0;
        double body_scale = 0.0;
        double elbow_angle_degrees = 0.0;
    };

    SideArm makeArm(const Pose* pose, JointId shoulder_id, JointId elbow_id,
                    JointId wrist_id, JointId hip_id) const {
        SideArm arm;
        if (pose == nullptr || !pose->valid) {
            return arm;
        }
        const auto& shoulder = pose->joint(shoulder_id);
        const auto& elbow = pose->joint(elbow_id);
        const auto& wrist = pose->joint(wrist_id);
        const auto& hip = pose->joint(hip_id);
        const auto usable = [this](const PoseKeypoint& point) {
            return point.valid &&
                   point.confidence >= config_.minimum_joint_confidence;
        };
        if (!usable(shoulder) || !usable(elbow) || !usable(wrist) ||
            !usable(hip)) {
            return arm;
        }
        arm.shoulder = shoulder.position_2d_pixels;
        arm.elbow = elbow.position_2d_pixels;
        arm.wrist = wrist.position_2d_pixels;
        arm.hip = hip.position_2d_pixels;
        arm.confidence =
            std::min({static_cast<double>(shoulder.confidence),
                      static_cast<double>(elbow.confidence),
                      static_cast<double>(wrist.confidence),
                      static_cast<double>(hip.confidence)});
        arm.body_scale =
            std::hypot(static_cast<double>(arm.shoulder.x - arm.hip.x),
                       static_cast<double>(arm.shoulder.y - arm.hip.y));
        arm.elbow_angle_degrees =
            angleDegrees(arm.shoulder, arm.elbow, arm.wrist);
        arm.valid = arm.body_scale > 1.0;
        return arm;
    }

    ArmObservation observe(const Pose* pose) const {
        const SideArm left =
            makeArm(pose, JointId::LeftShoulder, JointId::LeftElbow,
                    JointId::LeftWrist, JointId::LeftHip);
        const SideArm right =
            makeArm(pose, JointId::RightShoulder, JointId::RightElbow,
                    JointId::RightWrist, JointId::RightHip);

        if (filmSidePrefersLeftBodyChain(config_.film_side)) {
            return fromArm(left);
        }
        if (filmSidePrefersRightBodyChain(config_.film_side)) {
            return fromArm(right);
        }
        if (left.valid && right.valid) {
            ArmObservation result;
            result.valid = true;
            result.shoulder_y = (left.shoulder.y + right.shoulder.y) * 0.5;
            result.wrist_y = (left.wrist.y + right.wrist.y) * 0.5;
            result.body_scale = (left.body_scale + right.body_scale) * 0.5;
            result.elbow_angle_degrees =
                (left.elbow_angle_degrees + right.elbow_angle_degrees) * 0.5;
            return result;
        }
        return fromArm(left.valid ? left : right);
    }

    static ArmObservation fromArm(const SideArm& arm) {
        ArmObservation result;
        result.valid = arm.valid;
        if (arm.valid) {
            result.shoulder_y = arm.shoulder.y;
            result.wrist_y = arm.wrist.y;
            result.body_scale = arm.body_scale;
            result.elbow_angle_degrees = arm.elbow_angle_degrees;
        }
        return result;
    }

    static double smooth(double previous, double current, double alpha) {
        return alpha * current + (1.0 - alpha) * previous;
    }

    void beginRep(double source_time_seconds, double wrist_y, double body_scale,
                  double elbow_angle) {
        phase_ = OhpPhase::Pressing;
        rep_start_time_seconds_ = source_time_seconds;
        rep_start_wrist_y_ = wrist_y;
        highest_wrist_y_ = wrist_y;
        rep_body_scale_ = body_scale;
        maximum_elbow_angle_ = elbow_angle;
        speed_sum_ = 0.0;
        speed_samples_ = 0;
        peak_speed_ = 0.0;
    }

    void recordRepFrame(double wrist_y, double elbow_angle, double speed) {
        highest_wrist_y_ = std::min(highest_wrist_y_, wrist_y);
        maximum_elbow_angle_ = std::max(maximum_elbow_angle_, elbow_angle);
        const double magnitude = std::abs(speed);
        speed_sum_ += magnitude;
        ++speed_samples_;
        peak_speed_ = std::max(peak_speed_, magnitude);
    }

    void finishRep(double source_time_seconds, PoseAnalysisResult& result) {
        if (source_time_seconds - rep_start_time_seconds_ <
            config_.minimum_rep_duration_seconds) {
            phase_ = OhpPhase::Rack;
            return;
        }
        OhpRepSummary rep;
        rep.rep_index = static_cast<int>(completed_reps_.size()) + 1;
        rep.start_time_seconds = rep_start_time_seconds_;
        rep.end_time_seconds = source_time_seconds;
        rep.press_time_seconds =
            std::max(0.0, lockout_time_seconds_ - rep_start_time_seconds_);
        rep.lowering_time_seconds =
            std::max(0.0, source_time_seconds - lockout_time_seconds_);
        rep.maximum_elbow_angle_degrees = maximum_elbow_angle_;
        rep.wrist_travel_body_lengths =
            (rep_start_wrist_y_ - highest_wrist_y_) / rep_body_scale_;
        rep.average_normalized_speed_per_second =
            speed_samples_ == 0 ? 0.0 : speed_sum_ / speed_samples_;
        rep.peak_normalized_speed_per_second = peak_speed_;
        completed_reps_.push_back(rep);
        result.completed_ohp_rep = rep;
        phase_ = OhpPhase::Rack;
    }

    void updateState(double source_time_seconds, double wrist_y,
                     double body_scale, double elbow_angle, double speed,
                     PoseAnalysisResult& result) {
        const double threshold = config_.minimum_normalized_speed_per_second;
        const bool moving_up = speed < -threshold;
        const bool moving_down = speed > threshold;
        const bool in_rack =
            elbow_angle <= config_.rack_elbow_angle_degrees;

        if (phase_ == OhpPhase::Unknown) {
            if (in_rack) {
                phase_ = OhpPhase::Rack;
            }
            return;
        }

        if (phase_ == OhpPhase::Rack) {
            if (moving_up) {
                beginRep(source_time_seconds, previous_wrist_y_, body_scale,
                         elbow_angle);
                recordRepFrame(wrist_y, elbow_angle, speed);
            }
            return;
        }

        if (phase_ == OhpPhase::Pressing) {
            recordRepFrame(wrist_y, elbow_angle, speed);
            const double travel =
                (rep_start_wrist_y_ - highest_wrist_y_) / rep_body_scale_;
            if (elbow_angle >= config_.lockout_elbow_angle_degrees &&
                travel >= config_.minimum_wrist_travel_body_lengths) {
                lockout_time_seconds_ = source_time_seconds;
                phase_ = OhpPhase::Lockout;
            } else if (moving_down && travel <
                                          config_.minimum_wrist_travel_body_lengths) {
                phase_ = OhpPhase::Rack;
            }
            return;
        }

        if (phase_ == OhpPhase::Lockout) {
            recordRepFrame(wrist_y, elbow_angle, speed);
            if (moving_down ||
                elbow_angle < config_.lockout_exit_angle_degrees) {
                phase_ = OhpPhase::Lowering;
            }
            return;
        }

        if (phase_ == OhpPhase::Lowering) {
            recordRepFrame(wrist_y, elbow_angle, speed);
            const double return_distance =
                std::abs(wrist_y - rep_start_wrist_y_) / rep_body_scale_;
            if (in_rack &&
                return_distance <=
                    config_.rack_return_tolerance_body_lengths) {
                finishRep(source_time_seconds, result);
            }
        }
    }

    OhpAnalyzerConfig config_;
    OhpPhase phase_ = OhpPhase::Unknown;
    std::vector<OhpRepSummary> completed_reps_;
    std::size_t valid_frames_ = 0;
    std::size_t invalid_frames_ = 0;
    bool smoothed_wrist_valid_ = false;
    bool previous_wrist_valid_ = false;
    double smoothed_wrist_y_ = 0.0;
    double previous_wrist_y_ = 0.0;
    double previous_source_time_seconds_ = 0.0;
    double rep_start_time_seconds_ = 0.0;
    double rep_start_wrist_y_ = 0.0;
    double highest_wrist_y_ = 0.0;
    double rep_body_scale_ = 1.0;
    double lockout_time_seconds_ = 0.0;
    double maximum_elbow_angle_ = 0.0;
    double speed_sum_ = 0.0;
    std::size_t speed_samples_ = 0;
    double peak_speed_ = 0.0;
};

}  // namespace video_engine
