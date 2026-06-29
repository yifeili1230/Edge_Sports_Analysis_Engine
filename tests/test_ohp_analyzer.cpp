#include <cmath>
#include <iostream>
#include <vector>

#include "analytics/OhpAnalyzer.hpp"
#include "analytics/OhpSummaryWriter.hpp"
#include "processors/PoseAnalyticsProcessor.hpp"
#include "processors/PoseAnalyticsRenderer.hpp"

namespace {

video_engine::Pose makePose(const video_engine::Point2D& elbow,
                            const video_engine::Point2D& wrist,
                            float confidence = 0.95F) {
    video_engine::Pose pose;
    pose.valid = true;
    const video_engine::Point2D shoulder{100.0F, 150.0F};
    const video_engine::Point2D hip{100.0F, 250.0F};
    for (const auto [joint, point] :
         std::vector<std::pair<video_engine::JointId,
                               video_engine::Point2D>>{
             {video_engine::JointId::LeftShoulder, shoulder},
             {video_engine::JointId::LeftElbow, elbow},
             {video_engine::JointId::LeftWrist, wrist},
             {video_engine::JointId::LeftHip, hip},
             {video_engine::JointId::RightShoulder, shoulder},
             {video_engine::JointId::RightElbow, elbow},
             {video_engine::JointId::RightWrist, wrist},
             {video_engine::JointId::RightHip, hip},
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
    const double straight_angle = video_engine::OhpAnalyzer::angleDegrees(
        {100.0F, 150.0F}, {100.0F, 100.0F}, {100.0F, 50.0F});
    if (std::abs(straight_angle - 180.0) > 0.001) {
        std::cerr << "OHP elbow angle calculation is incorrect" << std::endl;
        return 1;
    }

    video_engine::OhpAnalyzerConfig config;
    config.smoothing_alpha = 1.0;
    config.minimum_rep_duration_seconds = 0.5;
    video_engine::OhpAnalyzer analyzer(config);

    const std::vector<video_engine::Point2D> elbows{
        {70, 150}, {70, 150}, {75, 140}, {80, 125}, {90, 110}, {100, 100},
        {100, 100}, {90, 110}, {80, 125}, {75, 140}, {70, 150}, {70, 150},
    };
    const std::vector<video_engine::Point2D> wrists{
        {70, 100}, {70, 100}, {75, 90}, {80, 75}, {90, 60}, {100, 50},
        {100, 50}, {90, 60}, {80, 75}, {75, 90}, {70, 100}, {70, 100},
    };

    video_engine::PoseAnalysisResult result;
    for (std::size_t index = 0; index < wrists.size(); ++index) {
        auto pose = makePose(elbows[index], wrists[index]);
        result = analyzer.analyze(&pose, index + 1, index * 0.1);
    }
    if (result.exercise != "ohp" ||
        result.ohp_phase != video_engine::OhpPhase::Rack ||
        result.completed_reps != 1 || analyzer.completedOhpReps().size() != 1 ||
        !result.completed_ohp_rep.has_value()) {
        std::cerr << "Complete rack-press-lockout-lower cycle was not counted"
                  << std::endl;
        return 1;
    }

    const auto& rep = analyzer.completedOhpReps().front();
    if (rep.maximum_elbow_angle_degrees < 170.0 ||
        rep.wrist_travel_body_lengths < 0.45 ||
        rep.press_time_seconds <= 0.0 ||
        rep.lowering_time_seconds <= 0.0 ||
        rep.peak_normalized_speed_per_second <= 0.0) {
        std::cerr << "Completed OHP rep metrics are incomplete" << std::endl;
        return 1;
    }

    video_engine::OhpAnalyzer incomplete(config);
    const std::vector<video_engine::Point2D> incomplete_elbows{
        {70, 150}, {75, 140}, {80, 125}, {75, 140}, {70, 150}};
    const std::vector<video_engine::Point2D> incomplete_wrists{
        {70, 100}, {75, 90}, {80, 75}, {75, 90}, {70, 100}};
    for (std::size_t index = 0; index < incomplete_wrists.size(); ++index) {
        auto pose =
            makePose(incomplete_elbows[index], incomplete_wrists[index]);
        incomplete.analyze(&pose, index + 1, index * 0.1);
    }
    if (!incomplete.completedOhpReps().empty()) {
        std::cerr << "OHP movement without lockout was counted" << std::endl;
        return 1;
    }

    auto invalid_pose =
        makePose({70, 150}, {70, 100}, 0.01F);
    const auto invalid = analyzer.analyze(&invalid_pose, 20, 2.0);
    if (invalid.valid || analyzer.invalidFrameCount() != 1 ||
        analyzer.completedOhpReps().size() != 1) {
        std::cerr << "Low-confidence observation corrupted OHP state"
                  << std::endl;
        return 1;
    }

    video_engine::OhpSessionSummary summary;
    summary.source = "video_source/ohp.mov";
    summary.processed_frames = wrists.size();
    summary.valid_analysis_frames = analyzer.validFrameCount();
    summary.invalid_analysis_frames = analyzer.invalidFrameCount();
    summary.reps = analyzer.completedOhpReps();
    const auto json = video_engine::OhpSummaryWriter::toJson(summary);
    if (video_engine::OhpSummaryWriter::outputPathForSource(
            summary.source, "output") !=
            std::filesystem::path("output/ohp.json") ||
        json.find("\"exercise\": \"ohp\"") == std::string::npos ||
        json.find("\"total_reps\": 1") == std::string::npos ||
        json.find("\"maximum_elbow_angle_degrees\"") ==
            std::string::npos) {
        std::cerr << "Input-named OHP JSON summary is incorrect" << std::endl;
        return 1;
    }

    video_engine::FrameContext context;
    context.processed_frame = cv::Mat::zeros(640, 640, CV_8UC3);
    context.pose_analysis = result;
    video_engine::PoseAnalyticsRenderer renderer;
    renderer.process(context);
    if (context.processed_frame.cols != 940 ||
        context.processed_frame.rows != 640) {
        std::cerr << "OHP analytics panel was not rendered" << std::endl;
        return 1;
    }

    return 0;
}
