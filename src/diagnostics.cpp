#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace {

bool buildFlagEnabled(const std::string& build_information, const std::string& name) {
    const auto position = build_information.find(name);
    if (position == std::string::npos) {
        return false;
    }
    const auto line_end = build_information.find('\n', position);
    const auto line = build_information.substr(position, line_end - position);
    return line.find("YES") != std::string::npos;
}

bool hasTarget(const std::vector<cv::dnn::Target>& targets, cv::dnn::Target wanted) {
    return std::find(targets.begin(), targets.end(), wanted) != targets.end();
}

}  // namespace

int main(int argc, char** argv) {
    const bool require_cuda =
        argc == 2 && std::string(argv[1]) == "--require-cuda";
    const std::string build_information = cv::getBuildInformation();
    const bool built_with_cuda = buildFlagEnabled(build_information, "NVIDIA CUDA:");
    const bool built_with_cudnn = buildFlagEnabled(build_information, "cuDNN:");

    std::vector<cv::dnn::Target> cuda_targets;
    try {
        cuda_targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
    } catch (const cv::Exception&) {
        // An OpenCV build without the CUDA DNN backend may reject this query.
    }

    const bool cuda_dnn =
        hasTarget(cuda_targets, cv::dnn::DNN_TARGET_CUDA);
    const bool cuda_fp16 =
        hasTarget(cuda_targets, cv::dnn::DNN_TARGET_CUDA_FP16);

    std::cout << "OpenCV version: " << CV_VERSION << '\n'
              << "CUDA compiled into OpenCV: " << (built_with_cuda ? "yes" : "no") << '\n'
              << "cuDNN compiled into OpenCV: " << (built_with_cudnn ? "yes" : "no") << '\n'
              << "DNN CUDA target available: " << (cuda_dnn ? "yes" : "no") << '\n'
              << "DNN CUDA FP16 target available: " << (cuda_fp16 ? "yes" : "no") << '\n';

    if (!cuda_fp16) {
        std::cout << "Result: CPU inference is ready; configs/pose_jetson.yaml is not "
                     "usable with this OpenCV build.\n";
    } else {
        std::cout << "Result: Jetson CUDA FP16 inference is ready.\n";
    }

    if (require_cuda && !cuda_fp16) {
        return 2;
    }
    return 0;
}
