#include <cassert>
#include <string>

#include "utils/Profiler.hpp"

int main() {
    video_engine::BenchmarkAccumulator benchmarks;
    benchmarks.addFrame({"resize", "pose_inference", "pipeline"},
                        {1.0, 30.0, 40.0});
    benchmarks.addFrame({"resize", "pose_inference", "pipeline"},
                        {3.0, 34.0, 44.0});

    const std::string summary =
        benchmarks.formatSummary("output/ohp.json", 2, 0.1);

    assert(summary ==
           "[Average Benchmarks] result=\"output/ohp.json\" frames=2 "
           "FPS: 20.0 | resize: 2.00 ms | pose_inference: 32.00 ms | "
           "pipeline: 42.00 ms");
    return 0;
}
