#include <opencv2/core.hpp>  // Provides cv::Mat and cv::Point for synthetic pose data.

#include "core/FrameContext.hpp"          // Provides pose storage for the renderer.
#include "processors/SkeletonRenderer.hpp" // Provides the skeleton renderer under test.

int main() {
    video_engine::FrameContext ctx;
    ctx.processed_frame = cv::Mat::zeros(240, 320, CV_8UC3);

    video_engine::Pose pose;
    pose.valid = true;
    const int points[] = {0, 1, 2, 3, 4, 8, 9, 10};
    for (int index : points) {
        pose.joints[index].valid = true;
        pose.joints[index].confidence = 0.9F;
        pose.joints[index].position_2d_pixels =
            video_engine::Point2D{static_cast<float>(80 + index * 8),
                                  static_cast<float>(60 + index * 4)};
    }
    ctx.poses.push_back(pose);

    video_engine::SkeletonRenderer renderer;
    renderer.process(ctx);

    return ctx.processed_frame.empty() ? 1 : 0;
}
