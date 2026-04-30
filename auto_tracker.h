#ifndef AUTO_TRACKER_H
#define AUTO_TRACKER_H

#include <opencv2/core.hpp>

class AutoTracker
{
public:
    static void processPixelCenter(const cv::Point& center, const cv::Size& frameSize);
    static void processPixelCenter(const cv::Point& center);
};

#endif // AUTO_TRACKER_H
