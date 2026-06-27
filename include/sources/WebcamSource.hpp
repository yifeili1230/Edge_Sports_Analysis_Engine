#pragma once

#include <chrono>  // Produces monotonic live-capture timestamps.

#include <opencv2/videoio.hpp>  // Provides cv::VideoCapture for camera input.

#include "core/IVideoSource.hpp"  // Defines the common video source interface.

namespace video_engine {

class WebcamSource : public IVideoSource {
public:
    explicit WebcamSource(int device = 0) : device_(device) {}

    bool open() override {
        capture_.open(device_, cv::CAP_ANY);
        if (!capture_.isOpened()) {
            return false;
        }
        start_time_ = std::chrono::steady_clock::now();
        last_timestamp_seconds_ = 0.0;
        return true;
    }

    bool read(cv::Mat& frame) override {
        if (!capture_.read(frame)) {
            return false;
        }
        last_timestamp_seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
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
    int device_;
    std::chrono::steady_clock::time_point start_time_;
    double last_timestamp_seconds_ = 0.0;
};

}  // namespace video_engine
