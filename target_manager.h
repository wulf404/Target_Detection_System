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
        bool externalLinkFresh = false;
        bool externalNoTarget = false;
        bool cameraBoxValid = false;
        int cameraBoxX = 0;
        int cameraBoxY = 0;
        int cameraBoxW = 0;
        int cameraBoxH = 0;
        int cameraFrameW = 0;
        int cameraFrameH = 0;
        int externalAzCentideg = 0;
        int externalElCentideg = 0;
        uint64_t cameraBoxSequence = 0;
        uint64_t cameraBoxLastSeenMs = 0;
        uint64_t externalLastSeenMs = 0;
        uint64_t externalLastPacketMs = 0;
    };

    static void submitCameraTarget(const cv::Point& center, const cv::Size& frameSize);
    static void submitCameraTarget(const cv::Point& center, const cv::Rect& targetBox, const cv::Size& frameSize);
    static void submitCameraMiss();
    static void submitExternalAnglesCentideg(int az_centideg, int el_centideg, bool valid);
    static void refresh();

    static Snapshot snapshot();
};
