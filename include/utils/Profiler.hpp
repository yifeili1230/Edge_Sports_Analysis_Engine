#pragma once

#include <algorithm> // Finds accumulated benchmark stages by name.
#include <chrono>   // Measures frame and stage timing.
#include <iomanip>  // Formats FPS and latency values.
#include <iostream> // Prints profiler output to stdout.
#include <sstream>  // Builds formatted profiler log lines.
#include <string>   // Stores stage names.
#include <vector>   // Stores stage latency arrays.

namespace video_engine {

class Profiler {
public:
    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsedMs(std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static void logFrameStats(size_t frame_index,
                              double fps,
                              const std::vector<std::string>& stage_names,
                              const std::vector<double>& latencies_ms) {
        std::ostringstream oss;
        oss << "[Frame " << frame_index << "] FPS: " << std::fixed << std::setprecision(1) << fps;
        for (size_t i = 0; i < stage_names.size() && i < latencies_ms.size(); ++i) {
            oss << " | " << stage_names[i] << ": " << std::setprecision(2) << latencies_ms[i] << " ms";
        }
        std::cout << oss.str() << std::endl;
    }
};

class BenchmarkAccumulator {
public:
    void addFrame(const std::vector<std::string>& stage_names,
                  const std::vector<double>& stage_latencies_ms) {
        const std::size_t stage_count =
            std::min(stage_names.size(), stage_latencies_ms.size());
        for (std::size_t index = 0; index < stage_count; ++index) {
            auto stage = std::find_if(
                stages_.begin(), stages_.end(),
                [&stage_names, index](const StageTotal& candidate) {
                    return candidate.name == stage_names[index];
                });
            if (stage == stages_.end()) {
                stages_.push_back(
                    {stage_names[index], stage_latencies_ms[index], 1});
                continue;
            }
            stage->latency_sum_ms += stage_latencies_ms[index];
            ++stage->sample_count;
        }
    }

    std::string formatSummary(const std::string& result_path,
                              std::size_t frame_count,
                              double elapsed_seconds) const {
        const double average_fps =
            elapsed_seconds > 0.0 ? frame_count / elapsed_seconds : 0.0;
        std::ostringstream output;
        output << "[Average Benchmarks] result=\"" << result_path << "\""
               << " frames=" << frame_count << " FPS: " << std::fixed
               << std::setprecision(1) << average_fps;
        for (const auto& stage : stages_) {
            const double average_latency_ms =
                stage.sample_count > 0
                    ? stage.latency_sum_ms / stage.sample_count
                    : 0.0;
            output << " | " << stage.name << ": " << std::setprecision(2)
                   << average_latency_ms << " ms";
        }
        return output.str();
    }

private:
    struct StageTotal {
        std::string name;
        double latency_sum_ms = 0.0;
        std::size_t sample_count = 0;
    };

    std::vector<StageTotal> stages_;
};

}  // namespace video_engine
