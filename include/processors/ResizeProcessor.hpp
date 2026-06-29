#pragma once

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>  // Provides cv::resize.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.

namespace video_engine {

class ResizeProcessor : public IFrameProcessor {
public:
    explicit ResizeProcessor(int width = 640, int height = 480,
                             bool preserve_aspect_ratio = true)
        : width_(width),
          height_(height),
          preserve_aspect_ratio_(preserve_aspect_ratio) {
        if (width_ <= 0 || height_ <= 0) {
            throw std::invalid_argument("Resize dimensions must be positive.");
        }
    }

    void process(FrameContext& ctx) override {
        if (ctx.raw_frame.empty()) {
            return;
        }
        cv::Size output_size(width_, height_);
        if (preserve_aspect_ratio_) {
            const double scale = std::min(
                width_ / static_cast<double>(ctx.raw_frame.cols),
                height_ / static_cast<double>(ctx.raw_frame.rows));
            output_size.width =
                std::max(1, cvRound(ctx.raw_frame.cols * scale));
            output_size.height =
                std::max(1, cvRound(ctx.raw_frame.rows * scale));
        }
        cv::resize(ctx.raw_frame, ctx.processed_frame, output_size);
    }

    std::string name() const override {
        return "resize";
    }

private:
    int width_;
    int height_;
    bool preserve_aspect_ratio_;
};

}  // namespace video_engine
