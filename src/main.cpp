#include <chrono>    // Provides timing utilities for FPS and pipeline latency measurement.
#include <cstdlib>   // Reads the optional analytics trace environment variable.
#include <csignal>   // Handles Ctrl+C as a graceful request to finish and save outputs.
#include <filesystem> // Provides output directory creation and file existence checks.
#include <fstream>   // Provides file input for reading configs/*.yaml.
#include <iostream>  // Provides standard output streams for error messages.
#include <memory>    // Provides std::unique_ptr for owned source, pipeline, and processor objects.
#include <stdexcept> // Provides runtime errors for invalid inference platform selections.
#include <string>    // Provides std::string for paths and backend names.
#include <vector>    // Provides std::vector for profiling stages.

#include <opencv2/highgui.hpp>  // Provides OpenCV window display APIs such as cv::imshow and cv::waitKey.
#include <opencv2/imgproc.hpp>  // Provides OpenCV image processing APIs such as resize and drawing.
#include <opencv2/videoio.hpp>  // Provides OpenCV video capture/output APIs such as cv::VideoWriter.

#include "analytics/IPoseAnalyzer.hpp"        // Defines the model-independent analytics interface.
#include "analytics/FilmSide.hpp"             // Defines the shared camera-facing body side.
#include "analytics/OhpAnalyzer.hpp"          // Implements overhead-press analysis.
#include "analytics/OhpSummaryWriter.hpp"     // Writes overhead-press JSON summaries.
#include "analytics/SquatAnalyzer.hpp"        // Implements mathematical squat analysis.
#include "analytics/SquatSummaryWriter.hpp"   // Writes the final input-named JSON summary.
#include "core/FrameContext.hpp"             // Defines the per-frame data object shared across processors.
#include "core/FrameTimeline.hpp"            // Assigns monotonic frame IDs and source timestamps.
#include "core/Pipeline.hpp"                 // Runs a sequence of IFrameProcessor stages.
#include "processors/PoseAnalyticsProcessor.hpp" // Connects canonical poses to exercise analysis.
#include "processors/PoseAnalyticsRenderer.hpp"  // Draws live squat metrics beside the video.
#include "processors/PoseEstimator.hpp"      // Runs human-pose keypoint inference with OpenCV DNN.
#include "processors/ResizeProcessor.hpp"    // Resizes each input frame to the configured dimensions.
#include "processors/SkeletonRenderer.hpp"   // Draws pose keypoints and skeleton connections.
#include "sources/VideoFileSource.hpp"       // Provides video-file input.
#include "sources/WebcamSource.hpp"          // Provides webcam input.
#include "utils/Profiler.hpp"                // Logs frame timing, FPS, and stage latency.

namespace {  // Keeps helper types and functions private to this translation unit.

volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int) {
    stop_requested = 1;
}

struct AppConfig {  // Stores all runtime configuration with defaults.
    std::string source = "webcam";  // Input source; "webcam" means camera, otherwise it is a file path.
    std::string config_path = "configs/pose.yaml";  // Default config file path.
    bool display = true;  // Whether to show a live OpenCV window.
    bool save_output = true;  // Whether to save processed frames as a video.
    std::string save_path = "output/output";  // Output video prefix used for numbered MP4 files.
    std::string exercise = "none";  // Optional exercise analyzer: none, squat, or ohp.
    std::string analysis_output_dir = "output";  // Directory for input-named analytics summaries.
    int width = 640;  // Processing frame width used by ResizeProcessor.
    int height = 480;  // Processing frame height used by ResizeProcessor.
    std::string resize_mode = "fit";  // "fit" preserves source geometry; "stretch" fills the exact size.
    std::string pose_model = "models/pose_iter_440000.caffemodel";  // Pose model weights path.
    std::string pose_config = "models/pose_deploy_linevec.prototxt";  // Pose model network definition path.
    int pose_input_width = 368;  // Width of the tensor sent into the pose model.
    int pose_input_height = 368;  // Height of the tensor sent into the pose model.
    float pose_confidence = 0.12F;  // Minimum keypoint confidence required for rendering.
    std::string pose_format = "openpose_coco";  // Model output contract: OpenPose COCO or YOLO COCO-17.
    float pose_detection_confidence = 0.25F;  // Minimum YOLO person-box confidence.
    float pose_nms_threshold = 0.45F;  // YOLO person-box overlap suppression threshold.
    int pose_max_people = 10;  // Maximum YOLO poses retained after NMS.
    std::string inference_platform = "manual";  // Optional profile that selects a matching DNN backend and target.
    std::string pose_backend = "opencv";  // OpenCV DNN backend, for example opencv, openvino, or cuda.
    std::string pose_target = "cpu";  // OpenCV DNN target device, for example cpu, opencl, cuda, or npu.
    double squat_standing_angle = 160.0;  // Knee angle that completes a return to standing.
    double squat_descent_start_angle = 155.0;  // Hysteresis threshold for leaving standing.
    double squat_descent_commit_angle = 145.0;
    double squat_bottom_angle = 100.0;  // Knee angle required to establish squat depth.
    double squat_bottom_exit_angle = 105.0;  // Hysteresis threshold for leaving the bottom.
    double squat_minimum_speed = 0.02;  // Minimum normalized vertical movement used for direction.
    double squat_direction_confirmation_speed = 0.50;
    double squat_smoothing_alpha = 0.35;  // Exponential smoothing weight for new observations.
    int squat_transition_confirmation_frames = 2;
    int squat_bottom_confirmation_frames = 1;
    double squat_minimum_phase_duration = 0.10;
    double squat_minimum_rep_duration = 0.50;
    double squat_maximum_angular_velocity = 720.0;
    double squat_maximum_normalized_speed = 4.0;
    double ohp_rack_elbow_angle = 120.0;
    double ohp_lockout_elbow_angle = 155.0;
    double ohp_lockout_exit_angle = 145.0;
    double ohp_minimum_speed = 0.04;
    double ohp_smoothing_alpha = 0.35;
    double ohp_minimum_wrist_travel = 0.35;
    double ohp_rack_return_tolerance = 0.20;
    double ohp_minimum_rep_duration = 0.50;
    video_engine::FilmSide film_side = video_engine::FilmSide::RightSideView;
    bool show_help = false;
};  // End of AppConfig.

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --config <path>                 Configuration file\n"
        << "  --source webcam|<video>         Input source\n"
        << "  --inference-platform cpu|jetson|tensorrt Inference profile\n"
        << "  --exercise none|squat|ohp       Optional pose analytics\n"
        << "  --film-side <view>              Camera view around the subject\n"
        << "  --analysis-output <dir>         JSON summary directory\n"
        << "  --display | --no-display        Toggle the GUI window\n"
        << "  --save <prefix> | --no-save     Toggle annotated video output\n"
        << "  -h, --help                      Show this help\n";
}

void configureInferencePlatform(AppConfig& config, const std::string& platform) {  // Applies convenient hardware-specific inference defaults.
    if (platform.empty() || platform == "manual") {  // Keeps explicitly configured backend and target values.
        return;  // Leaves the detailed DNN settings unchanged.
    }  // End of manual configuration branch.
    if (platform == "cpu") {  // Selects the portable CPU profile.
        config.pose_backend = "opencv";  // Uses OpenCV's built-in DNN implementation.
        config.pose_target = "cpu";  // Runs network layers on the CPU.
        return;  // Finishes applying the CPU profile.
    }  // End of CPU profile branch.
    if (platform == "jetson" || platform == "orin-nano" ||
        platform == "jetson-orin-nano") {  // Selects NVIDIA Jetson Orin Nano GPU inference.
        config.pose_backend = "cuda";  // Uses OpenCV DNN's CUDA backend.
        config.pose_target = "cuda_fp16";  // Uses FP16 CUDA inference to reduce compute and memory traffic.
        return;  // Finishes applying the Jetson profile.
    }  // End of Jetson profile branch.
    if (platform == "tensorrt" || platform == "jetson-tensorrt") {
        config.pose_backend = "tensorrt";
        config.pose_target = "fp16";
        return;
    }
    throw std::runtime_error(  // Rejects misspelled profiles instead of unexpectedly running on the CPU.
        "Unsupported inference platform: " + platform +
        ". Use manual, cpu, jetson, jetson-orin-nano, or tensorrt.");
}  // End of configureInferencePlatform.

void applyConfigFile(AppConfig& config, const std::string& path) {  // Applies config-file values to AppConfig.
    std::ifstream input(path);  // Opens the config file.
    if (!input.is_open()) {  // Handles missing or unreadable config files.
        throw std::runtime_error("Could not open configuration file: " + path);
    }  // End of open failure handling.

    std::string line;  // Holds the current config line.
    while (std::getline(input, line)) {  // Reads the config file one line at a time.
        auto comment = line.find('#');  // Finds an inline YAML-style comment marker.
        if (comment != std::string::npos) {  // Checks whether the line contains a comment.
            line = line.substr(0, comment);  // Removes the comment portion.
        }  // End of comment trimming.
        if (line.empty()) {  // Skips empty lines.
            continue;  // Moves to the next line.
        }  // End of empty-line handling.
        auto delimiter = line.find(':');  // Finds the key/value delimiter.
        if (delimiter == std::string::npos) {  // Skips lines without a delimiter.
            continue;  // Moves to the next line.
        }  // End of malformed-line handling.
        std::string key = line.substr(0, delimiter);  // Extracts the key on the left side of ':'.
        std::string value = line.substr(delimiter + 1);  // Extracts the value on the right side of ':'.
        key.erase(0, key.find_first_not_of(" \t"));  // Trims leading whitespace from the key.
        key.erase(key.find_last_not_of(" \t") + 1);  // Trims trailing whitespace from the key.
        value.erase(0, value.find_first_not_of(" \t"));  // Trims leading whitespace from the value.
        value.erase(value.find_last_not_of(" \t") + 1);  // Trims trailing whitespace from the value.

        if (key == "source") {  // Configures the input source.
            config.source = value;  // Stores the source string.
        } else if (key == "width") {  // Configures processing width.
            config.width = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "height") {  // Configures processing height.
            config.height = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "resize_mode") {
            config.resize_mode = value;
        } else if (key == "display") {  // Configures live display.
            config.display = value == "true";  // Enables display only for the literal value "true".
        } else if (key == "save_output") {  // Configures output video saving.
            config.save_output = value == "true";  // Enables saving only for the literal value "true".
        } else if (key == "save_path") {  // Configures output video path.
            config.save_path = value;  // Stores the output path.
        } else if (key == "exercise") {  // Selects an optional exercise analyzer.
            config.exercise = value;  // Stores none or squat for validation after CLI overrides.
        } else if (key == "analysis_output_dir") {  // Configures summary output location.
            config.analysis_output_dir = value;  // Stores the analytics directory.
        } else if (key == "pose_model") {  // Configures pose model weights path.
            config.pose_model = value;  // Stores the model path.
        } else if (key == "pose_config") {  // Configures pose model definition path.
            config.pose_config = value;  // Stores the model config path.
        } else if (key == "pose_input_width") {  // Configures pose model input width.
            config.pose_input_width = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "pose_input_height") {  // Configures pose model input height.
            config.pose_input_height = std::stoi(value);  // Converts the value to an integer.
        } else if (key == "pose_confidence") {  // Configures pose keypoint confidence threshold.
            config.pose_confidence = std::stof(value);  // Converts the value to a float.
        } else if (key == "pose_format") {
            config.pose_format = value;
        } else if (key == "pose_detection_confidence") {
            config.pose_detection_confidence = std::stof(value);
        } else if (key == "pose_nms_threshold") {
            config.pose_nms_threshold = std::stof(value);
        } else if (key == "pose_max_people") {
            config.pose_max_people = std::stoi(value);
        } else if (key == "inference_platform") {  // Configures a hardware-specific inference profile.
            config.inference_platform = value;  // Stores the profile name for application after all overrides.
        } else if (key == "pose_backend") {  // Configures the OpenCV DNN backend.
            config.pose_backend = value;  // Stores the backend name.
        } else if (key == "pose_target") {  // Configures the OpenCV DNN target device.
            config.pose_target = value;  // Stores the target device name.
        } else if (key == "squat_standing_angle") {
            config.squat_standing_angle = std::stod(value);
        } else if (key == "squat_descent_start_angle") {
            config.squat_descent_start_angle = std::stod(value);
        } else if (key == "squat_descent_commit_angle") {
            config.squat_descent_commit_angle = std::stod(value);
        } else if (key == "squat_bottom_angle") {
            config.squat_bottom_angle = std::stod(value);
        } else if (key == "squat_bottom_exit_angle") {
            config.squat_bottom_exit_angle = std::stod(value);
        } else if (key == "squat_minimum_speed") {
            config.squat_minimum_speed = std::stod(value);
        } else if (key == "squat_direction_confirmation_speed") {
            config.squat_direction_confirmation_speed = std::stod(value);
        } else if (key == "squat_smoothing_alpha") {
            config.squat_smoothing_alpha = std::stod(value);
        } else if (key == "squat_transition_confirmation_frames") {
            config.squat_transition_confirmation_frames = std::stoi(value);
        } else if (key == "squat_bottom_confirmation_frames") {
            config.squat_bottom_confirmation_frames = std::stoi(value);
        } else if (key == "squat_minimum_phase_duration") {
            config.squat_minimum_phase_duration = std::stod(value);
        } else if (key == "squat_minimum_rep_duration") {
            config.squat_minimum_rep_duration = std::stod(value);
        } else if (key == "squat_maximum_angular_velocity") {
            config.squat_maximum_angular_velocity = std::stod(value);
        } else if (key == "squat_maximum_normalized_speed") {
            config.squat_maximum_normalized_speed = std::stod(value);
        } else if (key == "ohp_rack_elbow_angle") {
            config.ohp_rack_elbow_angle = std::stod(value);
        } else if (key == "ohp_lockout_elbow_angle") {
            config.ohp_lockout_elbow_angle = std::stod(value);
        } else if (key == "ohp_lockout_exit_angle") {
            config.ohp_lockout_exit_angle = std::stod(value);
        } else if (key == "ohp_minimum_speed") {
            config.ohp_minimum_speed = std::stod(value);
        } else if (key == "ohp_smoothing_alpha") {
            config.ohp_smoothing_alpha = std::stod(value);
        } else if (key == "ohp_minimum_wrist_travel") {
            config.ohp_minimum_wrist_travel = std::stod(value);
        } else if (key == "ohp_rack_return_tolerance") {
            config.ohp_rack_return_tolerance = std::stod(value);
        } else if (key == "ohp_minimum_rep_duration") {
            config.ohp_minimum_rep_duration = std::stod(value);
        } else if (key == "film_side") {
            config.film_side = video_engine::parseFilmSide(value);
        }  // End of config key dispatch.
    }  // End of config-file line loop.
}  // End of applyConfigFile.

AppConfig parseArgs(int argc, char** argv) {  // Parses CLI arguments and loads the selected config file.
    AppConfig config;  // Starts with default configuration.
    for (int i = 1; i < argc; ++i) {  // First pass only finds the requested config file.
        std::string arg = argv[i];  // Wraps the current argument in std::string for comparison.
        if (arg == "--config" && i + 1 < argc) {  // Handles --config before loading file settings.
            config.config_path = argv[++i];  // Consumes the next argument as the config path.
        } else if (arg == "--help" || arg == "-h") {
            config.show_help = true;
        }  // End of config-path pre-scan.
    }  // End of config-path pre-scan loop.
    if (config.show_help) {
        return config;
    }
    applyConfigFile(config, config.config_path);  // Applies config-file values before CLI overrides.

    for (int i = 1; i < argc; ++i) {  // Iterates over user-provided command-line arguments.
        std::string arg = argv[i];  // Wraps the current argument in std::string for comparison.
        if (arg == "--source" && i + 1 < argc) {  // Handles --source when a value follows it.
            config.source = argv[++i];  // Consumes the next argument as the input source.
        } else if (arg == "--config" && i + 1 < argc) {  // Handles --config when a value follows it.
            config.config_path = argv[++i];  // Consumes the next argument as the config path.
        } else if (arg == "--display") {  // Handles explicit live-display request.
            config.display = true;  // Enables display.
        } else if (arg == "--no-display") {  // Handles headless operation over SSH or without a desktop session.
            config.display = false;  // Disables OpenCV window creation.
        } else if (arg == "--inference-platform" && i + 1 < argc) {  // Handles a named inference hardware profile.
            config.inference_platform = argv[++i];  // Consumes manual, cpu, or jetson.
        } else if (arg == "--save" && i + 1 < argc) {  // Handles output video saving.
            config.save_output = true;  // Enables saving.
            config.save_path = argv[++i];  // Consumes the next argument as the output path.
        } else if (arg == "--no-save") {  // Handles explicit output-video disabling.
            config.save_output = false;  // Avoids encoding overhead when only live inference is needed.
        } else if (arg == "--exercise" && i + 1 < argc) {  // Selects exercise analysis.
            config.exercise = argv[++i];  // Consumes none or squat.
        } else if (arg == "--film-side" && i + 1 < argc) {
            config.film_side = video_engine::parseFilmSide(argv[++i]);
        } else if (arg == "--analysis-output" && i + 1 < argc) {  // Selects summary directory.
            config.analysis_output_dir = argv[++i];  // Consumes the output directory.
        } else if (arg == "--help" || arg == "-h") {
            config.show_help = true;
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::runtime_error("Unknown or incomplete option: " + arg);
        }  // End of CLI argument dispatch.
    }  // End of CLI argument loop.
    if (config.show_help) {
        return config;
    }
    if (config.resize_mode != "fit" && config.resize_mode != "stretch") {
        throw std::runtime_error(
            "Unsupported resize_mode: " + config.resize_mode +
            ". Use fit or stretch.");
    }
    if (config.pose_confidence < 0.0F || config.pose_confidence > 1.0F ||
        config.pose_detection_confidence < 0.0F ||
        config.pose_detection_confidence > 1.0F ||
        config.pose_nms_threshold < 0.0F || config.pose_nms_threshold > 1.0F ||
        config.pose_max_people < 1) {
        throw std::runtime_error(
            "Pose confidence/NMS values must be in [0,1] and pose_max_people >= 1.");
    }
    if (config.exercise != "none" && config.exercise != "squat" &&
        config.exercise != "ohp") {
        throw std::runtime_error("Unsupported exercise: " + config.exercise +
                                 ". Use none, squat, or ohp.");
    }
    configureInferencePlatform(config, config.inference_platform);  // Resolves the final platform into backend and target settings.
    return config;  // Returns the final runtime configuration.
}  // End of parseArgs.

std::unique_ptr<video_engine::IVideoSource> createSource(const AppConfig& config) {  // Creates the video input source.
    if (config.source == "webcam") {  // Selects the default webcam source.
        return std::make_unique<video_engine::WebcamSource>(0);  // Opens camera index 0.
    }  // End of webcam branch.
    return std::make_unique<video_engine::VideoFileSource>(config.source);  // Treats source as a video file path.
}  // End of createSource.

std::unique_ptr<video_engine::Pipeline> buildPosePipeline(
    const AppConfig& config,
    std::shared_ptr<video_engine::IPoseAnalyzer>& pose_analyzer) {
    auto pipeline = std::make_unique<video_engine::Pipeline>();  // Creates an empty processor chain.
    pipeline->addProcessor(std::make_unique<video_engine::ResizeProcessor>(
        config.width, config.height, config.resize_mode == "fit"));  // Preserves geometry unless exact stretching is requested.
    pipeline->addProcessor(std::make_unique<video_engine::PoseEstimator>(
        config.pose_model, config.pose_config, config.pose_input_width,
        config.pose_input_height, config.pose_confidence, config.pose_backend,
        config.pose_target, config.pose_format, config.pose_detection_confidence,
        config.pose_nms_threshold, config.pose_max_people));
    if (config.exercise == "squat") {
        video_engine::SquatAnalyzerConfig analyzer_config;
        analyzer_config.minimum_joint_confidence = config.pose_confidence;
        analyzer_config.film_side = config.film_side;
        analyzer_config.standing_angle_degrees = config.squat_standing_angle;
        analyzer_config.descent_start_angle_degrees =
            config.squat_descent_start_angle;
        analyzer_config.descent_commit_angle_degrees =
            config.squat_descent_commit_angle;
        analyzer_config.bottom_angle_degrees = config.squat_bottom_angle;
        analyzer_config.bottom_exit_angle_degrees = config.squat_bottom_exit_angle;
        analyzer_config.minimum_normalized_speed_per_second =
            config.squat_minimum_speed;
        analyzer_config.direction_confirmation_speed_per_second =
            config.squat_direction_confirmation_speed;
        analyzer_config.smoothing_alpha = config.squat_smoothing_alpha;
        analyzer_config.transition_confirmation_frames =
            config.squat_transition_confirmation_frames;
        analyzer_config.bottom_confirmation_frames =
            config.squat_bottom_confirmation_frames;
        analyzer_config.minimum_phase_duration_seconds =
            config.squat_minimum_phase_duration;
        analyzer_config.minimum_rep_duration_seconds =
            config.squat_minimum_rep_duration;
        analyzer_config.maximum_knee_angular_velocity_degrees_per_second =
            config.squat_maximum_angular_velocity;
        analyzer_config.maximum_normalized_speed_per_second =
            config.squat_maximum_normalized_speed;
        pose_analyzer = std::make_shared<video_engine::SquatAnalyzer>(analyzer_config);
        pipeline->addProcessor(
            std::make_unique<video_engine::PoseAnalyticsProcessor>(pose_analyzer));
    } else if (config.exercise == "ohp") {
        video_engine::OhpAnalyzerConfig analyzer_config;
        analyzer_config.minimum_joint_confidence = config.pose_confidence;
        analyzer_config.film_side = config.film_side;
        analyzer_config.rack_elbow_angle_degrees =
            config.ohp_rack_elbow_angle;
        analyzer_config.lockout_elbow_angle_degrees =
            config.ohp_lockout_elbow_angle;
        analyzer_config.lockout_exit_angle_degrees =
            config.ohp_lockout_exit_angle;
        analyzer_config.minimum_normalized_speed_per_second =
            config.ohp_minimum_speed;
        analyzer_config.smoothing_alpha = config.ohp_smoothing_alpha;
        analyzer_config.minimum_wrist_travel_body_lengths =
            config.ohp_minimum_wrist_travel;
        analyzer_config.rack_return_tolerance_body_lengths =
            config.ohp_rack_return_tolerance;
        analyzer_config.minimum_rep_duration_seconds =
            config.ohp_minimum_rep_duration;
        pose_analyzer =
            std::make_shared<video_engine::OhpAnalyzer>(analyzer_config);
        pipeline->addProcessor(
            std::make_unique<video_engine::PoseAnalyticsProcessor>(pose_analyzer));
    }
    pipeline->addProcessor(std::make_unique<video_engine::SkeletonRenderer>());
    if (config.exercise == "squat" || config.exercise == "ohp") {
        pipeline->addProcessor(
            std::make_unique<video_engine::PoseAnalyticsRenderer>());
    }
    return pipeline;
}  // End of buildPosePipeline.

std::string nextOutputPath(const std::string& output_prefix) {  // Finds the first available numbered MP4 path.
    std::filesystem::path prefix(output_prefix);  // Treats the config value as a path prefix.
    std::filesystem::path directory = prefix.parent_path();  // Extracts the output directory.
    if (!directory.empty()) {  // Creates the directory only when the prefix includes one.
        std::filesystem::create_directories(directory);  // Ensures the output folder exists.
    }  // End of output directory setup.

    int index = 0;  // Starts numbering at output0.mp4.
    while (true) {  // Keeps trying names until an unused path is found.
        std::filesystem::path candidate = prefix.string() + std::to_string(index) + ".mp4";  // Builds outputN.mp4.
        if (!std::filesystem::exists(candidate)) {  // Uses this path only if it does not already exist.
            return candidate.string();  // Returns the first available output path.
        }  // End of availability check.
        ++index;  // Tries the next numbered filename.
    }  // End of output path search loop.
}  // End of nextOutputPath.

}  // namespace

int main(int argc, char** argv) {  // Program entry point.
    std::signal(SIGINT, requestStop);  // Lets Ctrl+C leave the loop and finalize video/JSON outputs.
    AppConfig config;  // Holds the final runtime configuration.
    try {
        config = parseArgs(argc, argv);  // Builds the runtime config from CLI and file input.
    } catch (const std::exception& error) {
        std::cerr << "Invalid configuration: " << error.what() << std::endl;  // Reports malformed values or unsupported profiles.
        return 1;  // Stops before opening video devices with an invalid configuration.
    }
    if (config.show_help) {
        printUsage(argv[0]);
        return 0;
    }
    if (config.pose_format != "yolo11_pose" &&
        config.pose_format != "yolo_coco17" &&
        !std::filesystem::is_regular_file(config.pose_config)) {
        std::cerr << "Pose network definition not found: " << config.pose_config
                  << std::endl;
        return 1;
    }
    if (!std::filesystem::is_regular_file(config.pose_model)) {
        std::cerr << "Pose model weights not found: " << config.pose_model << "\n"
                  << "Run scripts/download_pose_model.sh from the project root."
                  << std::endl;
        return 1;
    }
    auto source = createSource(config);  // Creates a webcam or video-file source.
    if (!source->open()) {  // Opens the source and checks for failure.
        std::cerr << "Failed to open source: " << config.source << std::endl;  // Reports the failed source path/name.
        return 1;  // Returns non-zero to indicate startup failure.
    }  // End of source open check.

    cv::VideoWriter writer;  // Creates a video writer that starts closed.
    std::string output_video_path;
    if (config.save_output) {  // Enables output writing only when requested.
        output_video_path = nextOutputPath(config.save_path);  // Chooses output0/output1/... without overwriting.
    }  // End of writer setup.

    std::unique_ptr<video_engine::Pipeline> pipeline;  // Owns the configured frame-processing pipeline.
    std::shared_ptr<video_engine::IPoseAnalyzer> pose_analyzer;
    try {
        pipeline = buildPosePipeline(config, pose_analyzer);
    } catch (const std::exception& error) {
        std::cerr << "Failed to build pipeline: " << error.what() << std::endl;  // Reports model or backend setup failures cleanly.
        return 1;  // Stops before entering the frame loop with an invalid pipeline.
    }
    video_engine::FrameContext ctx;  // Reused per-frame data container passed through the pipeline.
    video_engine::FrameTimeline timeline;  // Owns the monotonic frame and source-time contract.

    size_t frame_index = 0;  // Counts processed frames for FPS calculation.
    std::optional<video_engine::SquatPhase> reported_squat_phase;
    std::optional<video_engine::OhpPhase> reported_ohp_phase;
    const bool trace_analytics_frames =
        std::getenv("VIDEO_ENGINE_TRACE_ANALYTICS") != nullptr;
    video_engine::BenchmarkAccumulator benchmark_accumulator;
    auto start_time = video_engine::Profiler::now();  // Captures the start time for average FPS.
    double elapsed_seconds = 0.0;
    while (!stop_requested) {  // Main frame loop; Ctrl+C requests a graceful stop.
        cv::Mat frame;  // Holds the raw frame read from the input source.
        if (!source->read(frame)) {  // Reads the next frame from the selected source.
            break;  // Stops when the source has no more frames or cannot provide one.
        }  // End of frame read check.
        try {
            timeline.beginFrame(ctx, source->timestampSeconds());  // Starts a fresh pose measurement for this source frame.
        } catch (const std::exception& error) {
            std::cerr << "Invalid frame timeline: " << error.what() << std::endl;
            return 1;
        }
        ctx.raw_frame = frame;  // Stores the raw frame for downstream processors.
        ctx.processed_frame.release();  // Clears the previous processed frame.
        ctx.stage_names.clear();  // Clears previous profiling stage names.
        ctx.stage_latencies_ms.clear();  // Clears previous profiling stage timings.

        auto stage_start = video_engine::Profiler::now();  // Records pipeline start time.
        try {
        pipeline->run(ctx);  // Runs all processors in order; this is the core frame-processing call.
        } catch (const cv::Exception& error) {
            std::cerr << "OpenCV pipeline failure: " << error.what() << "\n"
                      << "Check the selected backend with build/video_engine_diagnostics."
                      << std::endl;
            return 1;
        } catch (const std::exception& error) {
            std::cerr << "Pipeline failure: " << error.what() << std::endl;
            return 1;
        }
        auto stage_end = video_engine::Profiler::now();  // Records pipeline end time.
        ctx.stage_names.push_back("pipeline");  // Adds a single aggregate pipeline stage label.
        ctx.stage_latencies_ms.push_back(video_engine::Profiler::elapsedMs(stage_start, stage_end));  // Stores total pipeline latency.

        if (config.exercise == "squat" && ctx.pose_analysis.has_value() &&
            ctx.pose_analysis->valid &&
            (!reported_squat_phase.has_value() ||
             *reported_squat_phase != ctx.pose_analysis->phase)) {
            reported_squat_phase = ctx.pose_analysis->phase;
            std::cout << "[Squat] frame=" << ctx.frame_id
                      << " time=" << ctx.source_time_seconds
                      << " phase=" << video_engine::squatPhaseName(
                             ctx.pose_analysis->phase)
                      << " knee=" << ctx.pose_analysis->knee_angle_degrees
                      << " speed="
                      << ctx.pose_analysis->normalized_vertical_speed_per_second
                      << std::endl;
        }
        if (config.exercise == "ohp" && ctx.pose_analysis.has_value() &&
            ctx.pose_analysis->valid &&
            (!reported_ohp_phase.has_value() ||
             *reported_ohp_phase != ctx.pose_analysis->ohp_phase)) {
            reported_ohp_phase = ctx.pose_analysis->ohp_phase;
            std::cout << "[OHP] frame=" << ctx.frame_id
                      << " time=" << ctx.source_time_seconds
                      << " phase="
                      << video_engine::ohpPhaseName(
                             ctx.pose_analysis->ohp_phase)
                      << " elbow=" << ctx.pose_analysis->elbow_angle_degrees
                      << " speed="
                      << ctx.pose_analysis->normalized_vertical_speed_per_second
                      << std::endl;
        }
        if (trace_analytics_frames && ctx.pose_analysis.has_value()) {
            const std::string phase =
                config.exercise == "ohp"
                    ? video_engine::ohpPhaseName(ctx.pose_analysis->ohp_phase)
                    : video_engine::squatPhaseName(ctx.pose_analysis->phase);
            std::cout << "[AnalyticsFrame] frame=" << ctx.frame_id
                      << " time=" << ctx.source_time_seconds
                      << " valid=" << (ctx.pose_analysis->valid ? 1 : 0)
                      << " phase=" << phase
                      << " angle="
                      << (config.exercise == "ohp"
                              ? ctx.pose_analysis->elbow_angle_degrees
                              : ctx.pose_analysis->knee_angle_degrees)
                      << " speed="
                      << ctx.pose_analysis->normalized_vertical_speed_per_second
                      << std::endl;
        }

        if (ctx.processed_frame.empty()) {  // Ensures there is always a frame to show or write.
            ctx.processed_frame = frame.clone();  // Falls back to the original frame.
        }  // End of processed-frame fallback.

        bool escape_pressed = false;
        if (config.display) {  // Shows the processed frame when display is enabled.
            cv::imshow("Video Engine", ctx.processed_frame);  // Updates the OpenCV display window.
            if (cv::waitKey(1) == 27) {  // Pumps window events and checks for ESC.
                escape_pressed = true;  // Finishes writing this frame before exiting.
            }  // End of key handling.
        }  // End of display branch.

        if (config.save_output && !writer.isOpened()) {
            writer.open(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        source->fps(), ctx.processed_frame.size());
            if (!writer.isOpened()) {
                std::cerr << "Failed to open output video: " << output_video_path << std::endl;
                return 1;
            }
            std::cout << "Saving output video to: " << output_video_path << std::endl;
        }
        if (writer.isOpened()) {  // Writes output only when the writer opened successfully.
            writer.write(ctx.processed_frame);  // Appends the processed frame to the output video.
        }  // End of output writing branch.

        ++frame_index;  // Advances the processed frame count.
        benchmark_accumulator.addFrame(ctx.stage_names,
                                       ctx.stage_latencies_ms);
        auto now = video_engine::Profiler::now();  // Gets the current timestamp.
        elapsed_seconds = std::chrono::duration<double>(now - start_time).count();  // Computes elapsed seconds.
        double fps = elapsed_seconds > 0 ? frame_index / elapsed_seconds : 0.0;  // Computes average FPS since startup.
        video_engine::Profiler::logFrameStats(frame_index, fps, ctx.stage_names, ctx.stage_latencies_ms);  // Logs frame stats and latency.
        if (escape_pressed) {
            break;
        }
    }  // End of main frame loop.

    writer.release();  // Finalizes the MP4 before reporting completion or writing the summary.
    cv::destroyAllWindows();  // Closes any OpenCV windows.

    if (pose_analyzer) {
        if (config.exercise == "ohp") {
            video_engine::OhpSessionSummary summary;
            summary.source = config.source;
            summary.processed_frames = frame_index;
            summary.valid_analysis_frames = pose_analyzer->validFrameCount();
            summary.invalid_analysis_frames = pose_analyzer->invalidFrameCount();
            summary.reps = pose_analyzer->completedOhpReps();
            const auto summary_path =
                video_engine::OhpSummaryWriter::outputPathForSource(
                    config.source, config.analysis_output_dir);
            try {
                video_engine::OhpSummaryWriter::write(summary_path, summary);
            } catch (const std::exception& error) {
                std::cerr << error.what() << std::endl;
                return 1;
            }
            std::cout << "OHP summary saved to: " << summary_path << std::endl;
            std::cout << benchmark_accumulator.formatSummary(
                             summary_path.string(), frame_index,
                             elapsed_seconds)
                      << std::endl;
            return 0;
        }
        video_engine::SquatSessionSummary summary;
        summary.source = config.source;
        summary.processed_frames = frame_index;
        summary.valid_analysis_frames = pose_analyzer->validFrameCount();
        summary.invalid_analysis_frames = pose_analyzer->invalidFrameCount();
        summary.reps = pose_analyzer->completedReps();
        const auto summary_path = video_engine::SquatSummaryWriter::outputPathForSource(
            config.source, config.analysis_output_dir);
        try {
            video_engine::SquatSummaryWriter::write(summary_path, summary);
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
        std::cout << "Squat summary saved to: " << summary_path << std::endl;
        std::cout << benchmark_accumulator.formatSummary(
                         summary_path.string(), frame_index, elapsed_seconds)
                  << std::endl;
    }
    return 0;  // Returns success.
}  // End of main.
