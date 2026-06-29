#pragma once

#include <cstdint>
#include <vector>

#include "analytics/AnalyticsTypes.hpp"
#include "analytics/FilmSide.hpp"
#include "pose/PoseTypes.hpp"

namespace video_engine {

class IPoseAnalyzer {
public:
    virtual ~IPoseAnalyzer() = default;

    virtual PoseAnalysisResult analyze(const Pose* pose, std::uint64_t frame_id,
                                       double source_time_seconds) = 0;
    virtual FilmSide filmSide() const noexcept = 0;
    virtual const std::vector<SquatRepSummary>& completedReps() const = 0;
    virtual const std::vector<OhpRepSummary>& completedOhpReps() const {
        static const std::vector<OhpRepSummary> no_reps;
        return no_reps;
    }
    virtual std::size_t validFrameCount() const = 0;
    virtual std::size_t invalidFrameCount() const = 0;
};

}  // namespace video_engine
