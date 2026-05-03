#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

class TargetManager
{
public:
    enum class Source
    {
        None,
        Camera,
        External
    };

    struct Snapshot
    {
        Source activeSource = Source::None;
        bool cameraFresh = false;
        bool externalFresh = false;
        int externalAzCentideg = 0;
        int externalElCentideg = 0;
        uint64_t externalLastSeenMs = 0;
    };

    static void submitCameraTarget(const cv::Point& center, const cv::Size& frameSize);
    static void submitCameraMiss();
    static void submitExternalAnglesCentideg(int az_centideg, int el_centideg, bool valid);
    static void refresh();

    static Snapshot snapshot();
};
