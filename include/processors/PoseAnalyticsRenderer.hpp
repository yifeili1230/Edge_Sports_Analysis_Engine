#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>

#include "analytics/AnalyticsTypes.hpp"
#include "core/IFrameProcessor.hpp"

namespace video_engine {

class PoseAnalyticsRenderer : public IFrameProcessor {
public:
    explicit PoseAnalyticsRenderer(int panel_width = 300) : panel_width_(panel_width) {}

    void process(FrameContext& ctx) override {
        if (ctx.processed_frame.empty()) {
            return;
        }

        cv::Mat canvas(ctx.processed_frame.rows, ctx.processed_frame.cols + panel_width_,
                       ctx.processed_frame.type(), cv::Scalar(20, 23, 29));
        ctx.processed_frame.copyTo(
            canvas(cv::Rect(0, 0, ctx.processed_frame.cols, ctx.processed_frame.rows)));
        cv::Mat panel = canvas(
            cv::Rect(ctx.processed_frame.cols, 0, panel_width_, ctx.processed_frame.rows));

        const bool is_ohp =
            ctx.pose_analysis.has_value() &&
            ctx.pose_analysis->exercise == "ohp";
        drawText(panel, is_ohp ? "OHP ANALYSIS" : "SQUAT ANALYSIS", 32, 0.70,
                 cv::Scalar(245, 245, 245), 2);
        cv::line(panel, cv::Point(20, 48), cv::Point(panel_width_ - 20, 48),
                 cv::Scalar(65, 70, 80), 1, cv::LINE_AA);

        if (!ctx.pose_analysis.has_value()) {
            drawText(panel, "Waiting for analysis", 86, 0.55, cv::Scalar(150, 155, 165), 1);
            ctx.processed_frame = canvas;
            return;
        }

        const auto& analysis = *ctx.pose_analysis;
        const cv::Scalar phase_color =
            is_ohp ? phaseColor(analysis.ohp_phase)
                   : phaseColor(analysis.phase);
        drawLabel(panel, "PHASE", 82);
        drawText(panel,
                 is_ohp ? ohpPhaseName(analysis.ohp_phase)
                        : squatPhaseName(analysis.phase),
                 110, 0.72, phase_color, 2);

        drawLabel(panel, "REPETITIONS", 154);
        drawText(panel, std::to_string(analysis.completed_reps), 192, 1.15,
                 cv::Scalar(255, 220, 80), 2);

        drawLabel(panel, "LIVE METRICS", 234);
        if (analysis.valid) {
            if (is_ohp) {
                drawText(panel,
                         "Elbow angle " +
                             format(analysis.elbow_angle_degrees, 1) + " deg",
                         264, 0.55, cv::Scalar(235, 235, 235), 1);
                drawText(panel,
                         "Wrist height " +
                             format(analysis.wrist_height_body_lengths, 2) +
                             " body",
                         294, 0.52, cv::Scalar(235, 235, 235), 1);
                drawText(panel,
                         "Wrist speed  " +
                             format(analysis.normalized_vertical_speed_per_second,
                                    2) +
                             " body/s",
                         324, 0.50, cv::Scalar(235, 235, 235), 1);
            } else {
                drawText(panel, "Knee angle  " + format(analysis.knee_angle_degrees, 1) + " deg",
                         264, 0.55, cv::Scalar(235, 235, 235), 1);
                const std::string hip_value = analysis.hip_angle_valid
                                                  ? format(analysis.hip_angle_degrees, 1) + " deg"
                                                  : "n/a";
                drawText(panel, "Hip angle   " + hip_value, 294, 0.55,
                         cv::Scalar(235, 235, 235), 1);
                drawText(panel,
                         "Back angle  " + format(analysis.back_angle_degrees, 1) + " deg",
                         318, 0.55, cv::Scalar(235, 235, 235), 1);
                drawText(panel,
                         "Ankle angle " + format(analysis.ankle_angle_degrees, 1) + " deg",
                         342, 0.55, cv::Scalar(235, 235, 235), 1);
                drawText(panel,
                         "Shoulder speed " +
                             format(analysis.normalized_vertical_speed_per_second, 2) + " body/s",
                         366, 0.50, cv::Scalar(235, 235, 235), 1);
            }
            const std::string film_side =
                analysis.film_side.has_value()
                    ? filmSideName(*analysis.film_side)
                    : "none";
            drawText(panel, "Film side  " + film_side, 390, 0.50,
                     cv::Scalar(180, 190, 205), 1);
        } else {
            drawText(panel, "Pose confidence too low", 270, 0.50,
                     cv::Scalar(80, 170, 255), 1);
            drawText(panel, "State is paused", 300, 0.50, cv::Scalar(180, 190, 205), 1);
        }

        if (is_ohp && analysis.completed_ohp_rep.has_value()) {
            const auto& rep = *analysis.completed_ohp_rep;
            drawText(panel, "REP " + std::to_string(rep.rep_index) + " COMPLETE",
                     414, 0.62, cv::Scalar(80, 230, 130), 2);
            drawText(panel,
                     "Max elbow " +
                         format(rep.maximum_elbow_angle_degrees, 1) + " deg",
                     440, 0.50, cv::Scalar(220, 225, 230), 1);
            drawText(panel,
                     "Up / down " + format(rep.press_time_seconds, 2) + " / " +
                         format(rep.lowering_time_seconds, 2) + " s",
                     466, 0.48, cv::Scalar(220, 225, 230), 1);
        } else if (!is_ohp && analysis.completed_rep.has_value()) {
            const auto& rep = *analysis.completed_rep;
            drawText(panel, "REP " + std::to_string(rep.rep_index) + " COMPLETE", 414, 0.62,
                     cv::Scalar(80, 230, 130), 2);
            drawText(panel, "Min knee  " + format(rep.minimum_knee_angle_degrees, 1) + " deg",
                     440, 0.50, cv::Scalar(220, 225, 230), 1);
            drawText(panel, "Down / up  " + format(rep.descent_time_seconds, 2) + " / " +
                                format(rep.ascent_time_seconds, 2) + " s",
                     466, 0.48, cv::Scalar(220, 225, 230), 1);
        } else {
            drawText(panel, "ESC: finish and save summary", panel.rows - 22, 0.42,
                     cv::Scalar(125, 135, 150), 1);
        }

        ctx.processed_frame = canvas;
    }

    std::string name() const override {
        return "pose_analytics_renderer";
    }

private:
    static std::string format(double value, int precision) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }

    static cv::Scalar phaseColor(SquatPhase phase) {
        switch (phase) {
            case SquatPhase::Descending:
                return cv::Scalar(80, 190, 255);
            case SquatPhase::Bottom:
                return cv::Scalar(90, 90, 255);
            case SquatPhase::Ascending:
                return cv::Scalar(100, 230, 140);
            case SquatPhase::Standing:
                return cv::Scalar(255, 220, 80);
            case SquatPhase::Unknown:
            default:
                return cv::Scalar(160, 165, 175);
        }
    }

    static cv::Scalar phaseColor(OhpPhase phase) {
        switch (phase) {
            case OhpPhase::Pressing:
                return cv::Scalar(100, 230, 140);
            case OhpPhase::Lockout:
                return cv::Scalar(255, 220, 80);
            case OhpPhase::Lowering:
                return cv::Scalar(80, 190, 255);
            case OhpPhase::Rack:
                return cv::Scalar(210, 180, 100);
            case OhpPhase::Unknown:
            default:
                return cv::Scalar(160, 165, 175);
        }
    }

    static void drawLabel(cv::Mat& panel, const std::string& text, int y) {
        drawText(panel, text, y, 0.43, cv::Scalar(130, 140, 155), 1);
    }

    static void drawText(cv::Mat& panel, const std::string& text, int y, double scale,
                         const cv::Scalar& color, int thickness) {
        cv::putText(panel, text, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv::LINE_AA);
    }

    int panel_width_;
};

}  // namespace video_engine
