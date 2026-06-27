#pragma once

#include <opencv2/core.hpp>  // Provides cv::Mat for decoded video frames.

namespace video_engine {

class IVideoSource {
public:
    virtual ~IVideoSource() = default;
    virtual bool open() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual double fps() const = 0;
    virtual double timestampSeconds() const = 0;
};

}  // namespace video_engine
