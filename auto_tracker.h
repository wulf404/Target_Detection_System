#ifndef AUTO_TRACKER_H
#define AUTO_TRACKER_H

#include <opencv2/core.hpp>

class AutoTracker
{
public:
    struct OverlayConfig
    {
        int deadzoneX = 0;
        int deadzoneY = 0;
        int deadzoneOuterX = 0;
        int deadzoneOuterY = 0;
        bool deadzoneEnabled = false;
        bool deadzoneHoldActive = false;
    };

    static void processPixelCenter(const cv::Point& center, const cv::Size& frameSize);
    static void processPixelCenter(const cv::Point& center);
    static void reset();
    static OverlayConfig overlayConfig();
};

#endif // AUTO_TRACKER_H
