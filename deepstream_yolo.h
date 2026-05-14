#pragma once

#include <opencv2/core.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <mutex>
#include <string>
#include <vector>

class DeepStreamYolo
{
public:
    struct Detection
    {
        int classId = -1;
        float confidence = 0.0f;
        cv::Rect box;
        cv::Point center;
        int area = 0;
    };

    DeepStreamYolo();
    ~DeepStreamYolo();

    bool open(const std::string& devicePath, int width, int height, int fps);
    bool isOpened() const;
    bool read(cv::Mat& frame, bool trackingEnabled);
    void close();

    std::string statusText() const;

private:
    struct Candidate
    {
        int index = -1;
        Detection detection;
        double score = 0.0;
    };

    static GstPadProbeReturn tensorProbe(GstPad* pad, GstPadProbeInfo* info, gpointer userData);

    bool writeInferConfig() const;
    std::string buildPipeline(const std::string& devicePath, int width, int height, int fps) const;
    bool checkBusErrors();
    void updateDetectionsFromBuffer(GstBuffer* buffer);
    std::vector<Detection> parseTensorMeta(void* tensorMeta, int frameWidth, int frameHeight) const;
    void processDetections(cv::Mat& frame, const std::vector<Detection>& detections, bool trackingEnabled);
    void loadClasses();

    GstElement* pipeline = nullptr;
    GstElement* appsink = nullptr;

    mutable std::mutex detectionsMutex;
    std::vector<Detection> latestDetections;

    std::vector<std::string> classes;
    cv::Point lastTargetCenter;
    bool haveLastTarget = false;
    int lostTargetFrames = 0;

    std::string currentDevice;
    int requestedWidth = 0;
    int requestedHeight = 0;
    int requestedFps = 0;
    int actualWidth = 0;
    int actualHeight = 0;
};
