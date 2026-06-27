#pragma once

#include <memory>  // Owns processors through std::unique_ptr.
#include <vector>  // Stores the ordered processor chain.

#include "core/IFrameProcessor.hpp"  // Defines the processor interface run by the pipeline.
#include "utils/Profiler.hpp"        // Measures each processor's wall-clock latency.

namespace video_engine {

class Pipeline {
public:
    void addProcessor(std::unique_ptr<IFrameProcessor> processor) {
        processors_.push_back(std::move(processor));
    }

    void run(FrameContext& ctx) {
        for (const auto& processor : processors_) {
            const auto stage_start = Profiler::now();
            processor->process(ctx);
            const auto stage_end = Profiler::now();
            ctx.stage_names.push_back(processor->name());
            ctx.stage_latencies_ms.push_back(Profiler::elapsedMs(stage_start, stage_end));
        }
    }

private:
    std::vector<std::unique_ptr<IFrameProcessor>> processors_;
};

}  // namespace video_engine
