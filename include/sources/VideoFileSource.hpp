#pragma once

#include <cstdint> // Stores the number of decoded source frames.
#include <string>  // Stores the input video file path.

#include <opencv2/videoio.hpp>  // Provides cv::VideoCapture for video file input.

#include "core/IVideoSource.hpp"  // Defines the common video source interface.

namespace video_engine {

class VideoFileSource : public IVideoSource {
public:
    explicit VideoFileSource(std::string path) : path_(std::move(path)) {}

    bool open() override {
        capture_.open(path_);
        frames_read_ = 0;
        last_timestamp_seconds_ = 0.0;
        return capture_.isOpened();
    }

    bool read(cv::Mat& frame) override {
        if (!capture_.read(frame)) {
            return false;
        }

        const double fallback_time = static_cast<double>(frames_read_) / fps();
        const double media_time_ms = capture_.get(cv::CAP_PROP_POS_MSEC);
        double candidate_time = media_time_ms >= 0.0 ? media_time_ms / 1000.0 : fallback_time;
        if (frames_read_ > 0 && candidate_time <= last_timestamp_seconds_) {
            candidate_time = fallback_time;
        }
        if (frames_read_ > 0 && candidate_time <= last_timestamp_seconds_) {
            candidate_time = last_timestamp_seconds_ + 1.0 / fps();
        }

        last_timestamp_seconds_ = candidate_time;
        ++frames_read_;
        return true;
    }

    double fps() const override {
        return capture_.get(cv::CAP_PROP_FPS) > 0 ? capture_.get(cv::CAP_PROP_FPS) : 30.0;
    }

    double timestampSeconds() const override {
        return last_timestamp_seconds_;
    }

private:
    cv::VideoCapture capture_;
    std::string path_;
    std::uint64_t frames_read_ = 0;
    double last_timestamp_seconds_ = 0.0;
};

}  // namespace video_engine
