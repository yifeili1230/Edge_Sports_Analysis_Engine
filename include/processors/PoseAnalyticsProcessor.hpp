#pragma once

#include <memory>

#include "analytics/IPoseAnalyzer.hpp"
#include "core/IFrameProcessor.hpp"

namespace video_engine {

class PoseAnalyticsProcessor : public IFrameProcessor {
public:
    explicit PoseAnalyticsProcessor(std::shared_ptr<IPoseAnalyzer> analyzer)
        : analyzer_(std::move(analyzer)) {}

    void process(FrameContext& ctx) override {
        const Pose* pose = nullptr;
        if (ctx.pose_measurement_valid && !ctx.poses.empty()) {
            pose = &ctx.poses.front();
        }
        ctx.pose_analysis =
            analyzer_->analyze(pose, ctx.frame_id, ctx.source_time_seconds);
        ctx.completed_rep_events.clear();
        ctx.completed_ohp_rep_events.clear();
        if (ctx.pose_analysis->completed_rep.has_value()) {
            const auto& rep = *ctx.pose_analysis->completed_rep;
            if (rep.rep_index != last_emitted_rep_index_) {
                ctx.completed_rep_events.push_back(rep);
                last_emitted_rep_index_ = rep.rep_index;
            }
        }
        if (ctx.pose_analysis->completed_ohp_rep.has_value()) {
            const auto& rep = *ctx.pose_analysis->completed_ohp_rep;
            if (rep.rep_index != last_emitted_ohp_rep_index_) {
                ctx.completed_ohp_rep_events.push_back(rep);
                last_emitted_ohp_rep_index_ = rep.rep_index;
            }
        }
    }

    std::string name() const override {
        return "pose_analytics";
    }

private:
    std::shared_ptr<IPoseAnalyzer> analyzer_;
    int last_emitted_rep_index_ = 0;
    int last_emitted_ohp_rep_index_ = 0;
};

}  // namespace video_engine
