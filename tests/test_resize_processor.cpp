#include <iostream>

#include <opencv2/core.hpp>

#include "processors/ResizeProcessor.hpp"

int main() {
    video_engine::FrameContext context;
    context.raw_frame = cv::Mat::zeros(1920, 1080, CV_8UC3);

    video_engine::ResizeProcessor fit(640, 640, true);
    fit.process(context);
    if (context.processed_frame.cols != 360 ||
        context.processed_frame.rows != 640) {
        std::cerr << "Aspect-ratio fit distorted the 9:16 frame" << std::endl;
        return 1;
    }

    video_engine::ResizeProcessor stretch(640, 480, false);
    stretch.process(context);
    if (context.processed_frame.cols != 640 ||
        context.processed_frame.rows != 480) {
        std::cerr << "Explicit stretch did not produce the requested size" << std::endl;
        return 1;
    }
    return 0;
}
