#include <cmath>
#include <iostream>
#include <vector>

#include "analytics/SquatAnalyzer.hpp"
#include "analytics/SquatSummaryWriter.hpp"
#include "processors/PoseAnalyticsRenderer.hpp"

namespace {

video_engine::Pose makePose(double knee_angle_degrees, double hip_y,
                            float confidence = 0.95F) {
    video_engine::Pose pose;
    pose.valid = true;

    constexpr double kPi = 3.14159265358979323846;
    const double radians = (-90.0 + knee_angle_degrees) * kPi / 180.0;
    const video_engine::Point2D hip{100.0F, static_cast<float>(hip_y)};
    const video_engine::Point2D knee{100.0F, static_cast<float>(hip_y + 50.0)};
    const video_engine::Point2D ankle{
        static_cast<float>(knee.x + 50.0 * std::cos(radians)),
        static_cast<float>(knee.y + 50.0 * std::sin(radians)),
    };
    const video_engine::Point2D shoulder{100.0F, static_cast<float>(hip_y - 50.0)};

    for (const auto [joint, point] :
         std::vector<std::pair<video_engine::JointId, video_engine::Point2D>>{
             {video_engine::JointId::RightShoulder, shoulder},
             {video_engine::JointId::RightHip, hip},
             {video_engine::JointId::RightKnee, knee},
             {video_engine::JointId::RightAnkle, ankle},
         }) {
        auto& keypoint = pose.joint(joint);
        keypoint.position_2d_pixels = point;
        keypoint.confidence = confidence;
        keypoint.valid = true;
    }
    return pose;
}

}  // namespace

int main() {
    const auto right_angle = video_engine::SquatAnalyzer::angleDegrees(
        video_engine::Point2D{0.0F, 1.0F}, video_engine::Point2D{0.0F, 0.0F},
        video_engine::Point2D{1.0F, 0.0F});
    if (std::abs(right_angle - 90.0) > 0.001) {
        std::cerr << "2D joint angle calculation is incorrect" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzerConfig config;
    config.smoothing_alpha = 1.0;
    video_engine::SquatAnalyzer analyzer(config);
    if (analyzer.filmSide() != video_engine::FilmSide::RightSideView) {
        std::cerr << "Default film side is not exposed through IPoseAnalyzer" << std::endl;
        return 1;
    }
    video_engine::SquatAnalyzerConfig left_config = config;
    left_config.film_side = video_engine::FilmSide::LeftSideView;
    video_engine::SquatAnalyzer left_analyzer(left_config);
    video_engine::IPoseAnalyzer& analyzer_interface = left_analyzer;
    if (analyzer_interface.filmSide() != video_engine::FilmSide::LeftSideView ||
        std::string(video_engine::filmSideName(analyzer_interface.filmSide())) !=
            "left_side_view") {
        std::cerr << "Shared analyzer film-side interface is incorrect" << std::endl;
        return 1;
    }
    const std::vector<std::string> film_side_names{
        "front_view",       "front_left_view", "front_right_view",
        "left_side_view",   "right_side_view", "rear_left_view",
        "rear_view",        "rear_right_view",
    };
    for (const auto& name : film_side_names) {
        const auto side = video_engine::parseFilmSide(name);
        if (name != video_engine::filmSideName(side)) {
            std::cerr << "Film-side value did not round-trip: " << name << std::endl;
            return 1;
        }
    }
    video_engine::SquatAnalyzerConfig front_config = config;
    front_config.film_side = video_engine::FilmSide::FrontView;
    front_config.smoothing_alpha = 1.0;
    video_engine::SquatAnalyzer front_analyzer(front_config);
    auto front_pose = makePose(170.0, 50.0);
    const auto front_result = front_analyzer.analyze(&front_pose, 1, 0.0);
    if (!front_result.valid ||
        front_result.film_side != video_engine::FilmSide::FrontView) {
        std::cerr << "Front view did not select the available body chain" << std::endl;
        return 1;
    }
    const std::vector<double> angles{
        170.0, 166.0, 150.0, 130.0, 110.0, 95.0, 98.0, 110.0, 130.0, 150.0,
        165.0, 168.0,
    };
    const std::vector<double> hip_y{
        50.0, 50.0, 53.0, 58.0, 64.0, 70.0, 70.0, 67.0, 61.0, 55.0, 50.0,
        50.0,
    };

    video_engine::PoseAnalysisResult result;
    for (std::size_t index = 0; index < angles.size(); ++index) {
        auto pose = makePose(angles[index], hip_y[index]);
        result = analyzer.analyze(&pose, index + 1, index * 0.1);
    }

    if (analyzer.completedReps().size() != 1 || result.completed_reps != 1 ||
        !result.completed_rep.has_value() || result.completed_rep->rep_index != 1) {
        std::cerr << "Complete standing-descending-bottom-ascending cycle was not counted"
                  << " (phase=" << video_engine::squatPhaseName(result.phase)
                  << ", reps=" << analyzer.completedReps().size() << ")"
                  << std::endl;
        return 1;
    }

    if (!result.hip_angle_valid || !result.back_angle_valid ||
        !result.ankle_angle_valid ||
        std::abs(result.hip_angle_degrees - 180.0) > 0.01 ||
        std::abs(result.back_angle_degrees) > 0.01) {
        std::cerr << "Hip, back, knee, and ankle angles were not retained" << std::endl;
        return 1;
    }

    const auto& rep = analyzer.completedReps().front();
    if (rep.minimum_knee_angle_degrees > 100.0 || rep.descent_time_seconds <= 0.0 ||
        rep.ascent_time_seconds <= 0.0 || rep.peak_normalized_speed_per_second <= 0.0) {
        std::cerr << "Completed rep metrics are incomplete" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzer repeated_analyzer(config);
    video_engine::PoseAnalysisResult repeated_result;
    for (int repetition = 0; repetition < 2; ++repetition) {
        for (std::size_t index = 0; index < angles.size(); ++index) {
            auto pose = makePose(angles[index], hip_y[index]);
            const auto frame = repetition * angles.size() + index;
            repeated_result =
                repeated_analyzer.analyze(&pose, frame + 1, frame * 0.1);
            if (repetition == 1 && index > 1 && index < 10 &&
                (!repeated_result.completed_rep.has_value() ||
                 repeated_result.completed_rep->rep_index != 1)) {
                std::cerr << "Completed rep did not stay visible during the next rep"
                          << std::endl;
                return 1;
            }
        }
    }
    if (repeated_result.completed_reps != 2 ||
        !repeated_result.completed_rep.has_value() ||
        repeated_result.completed_rep->rep_index != 2) {
        std::cerr << "Persistent rep display did not update after the next rep"
                  << std::endl;
        return 1;
    }

    auto invalid_pose = makePose(90.0, 70.0, 0.01F);
    const auto invalid = analyzer.analyze(&invalid_pose, 20, 2.0);
    if (invalid.valid || analyzer.invalidFrameCount() != 1 ||
        analyzer.completedReps().size() != 1) {
        std::cerr << "Low-confidence observation corrupted squat state" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzer incomplete_analyzer(config);
    const std::vector<double> incomplete_angles{170.0, 150.0, 120.0, 110.0, 150.0, 170.0};
    for (std::size_t index = 0; index < incomplete_angles.size(); ++index) {
        auto pose = makePose(incomplete_angles[index], 50.0 + index);
        incomplete_analyzer.analyze(&pose, index + 1, index * 0.1);
    }
    if (!incomplete_analyzer.completedReps().empty()) {
        std::cerr << "Squat that never reached the bottom threshold was counted" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzer small_movement_analyzer(config);
    const std::vector<double> small_movement_angles{170.0, 150.0, 165.0, 168.0};
    video_engine::PoseAnalysisResult small_movement_result;
    for (std::size_t index = 0; index < small_movement_angles.size(); ++index) {
        auto pose = makePose(small_movement_angles[index], 50.0);
        small_movement_result =
            small_movement_analyzer.analyze(&pose, index + 1, index * 0.1);
    }
    if (small_movement_result.phase != video_engine::SquatPhase::Standing ||
        !small_movement_analyzer.completedReps().empty()) {
        std::cerr << "One-frame knee movement started a false squat" << std::endl;
        return 1;
    }

    video_engine::SquatAnalyzer speed_analyzer(config);
    auto standing_pose = makePose(170.0, 50.0);
    speed_analyzer.analyze(&standing_pose, 1, 0.0);
    auto lower_shoulder_pose = makePose(150.0, 55.0);
    const auto descending_result =
        speed_analyzer.analyze(&lower_shoulder_pose, 2, 0.1);
    if (!descending_result.valid ||
        descending_result.phase != video_engine::SquatPhase::Descending ||
        descending_result.normalized_vertical_speed_per_second <= 0.0) {
        std::cerr << "Shoulder descent did not produce downward speed" << std::endl;
        return 1;
    }

    video_engine::SquatSessionSummary summary;
    summary.source = "video_source/IMG_1389.mov";
    summary.processed_frames = angles.size();
    summary.valid_analysis_frames = analyzer.validFrameCount();
    summary.invalid_analysis_frames = analyzer.invalidFrameCount();
    summary.reps = analyzer.completedReps();
    const auto summary_path =
        video_engine::SquatSummaryWriter::outputPathForSource(summary.source, "output");
    const auto json = video_engine::SquatSummaryWriter::toJson(summary);
    if (summary_path != std::filesystem::path("output/IMG_1389.json") ||
        json.find("\"total_reps\": 1") == std::string::npos ||
        json.find("\"minimum_knee_angle_degrees\"") == std::string::npos) {
        std::cerr << "Input-named squat JSON summary is incorrect" << std::endl;
        return 1;
    }

    video_engine::FrameContext context;
    context.processed_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    context.pose_analysis = result;
    video_engine::PoseAnalyticsRenderer renderer;
    renderer.process(context);
    if (context.processed_frame.cols != 940 || context.processed_frame.rows != 480) {
        std::cerr << "Live analytics panel did not preserve video and add side panel"
                  << std::endl;
        return 1;
    }

    return 0;
}
