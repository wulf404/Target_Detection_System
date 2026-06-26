#include "my_yolo.h"
#include "auto_tracker.h"
#include "app_config.h"
#include "latency_monitor.h"
#include "target_manager.h"

#include <NvInferPlugin.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <chrono>

static inline double ms(const cv::TickMeter& tm) { return tm.getTimeMilli(); }

namespace {

struct TargetCandidate
{
    int index = -1;
    cv::Rect box;
    cv::Point center;
    int area = 0;
    float confidence = 0.0f;
    double score = 0.0;
    bool suspicious = false;
};

uint64_t monotonicMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

double clamp01(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

double hypotSize(const cv::Size2d& size)
{
    return std::hypot(size.width, size.height);
}

cv::Point roundPoint(const cv::Point2d& point)
{
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

cv::Point clampPointToFrame(const cv::Point& point, const cv::Size& frameSize)
{
    if (frameSize.width <= 0 || frameSize.height <= 0) {
        return point;
    }

    return cv::Point(
        std::clamp(point.x, 0, frameSize.width - 1),
        std::clamp(point.y, 0, frameSize.height - 1)
    );
}

cv::Rect rectFromCenterSize(const cv::Point2d& center, const cv::Size2d& size)
{
    const int w = std::max(1, cvRound(size.width));
    const int h = std::max(1, cvRound(size.height));
    return cv::Rect(cvRound(center.x - w * 0.5),
                    cvRound(center.y - h * 0.5),
                    w,
                    h);
}

int64_t dimsVolume(const nvinfer1::Dims& dims)
{
    int64_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        volume *= static_cast<int64_t>(dims.d[i]);
    }
    return volume;
}

size_t dataTypeSize(nvinfer1::DataType dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return sizeof(__half);
    case nvinfer1::DataType::kINT8:
        return sizeof(std::int8_t);
    case nvinfer1::DataType::kINT32:
        return sizeof(std::int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    default:
        return 0;
    }
}

std::string dimsToString(const nvinfer1::Dims& dims)
{
    std::ostringstream out;
    out << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            out << "x";
        }
        out << dims.d[i];
    }
    out << "]";
    return out.str();
}

void drawDashedLine(cv::Mat& frame,
                    const cv::Point& from,
                    const cv::Point& to,
                    const cv::Scalar& color,
                    int thickness = 2,
                    double dash_px = 18.0,
                    double gap_px = 10.0)
{
    const double dx = static_cast<double>(to.x - from.x);
    const double dy = static_cast<double>(to.y - from.y);
    const double length = std::hypot(dx, dy);
    if (length < 1.0) {
        return;
    }

    const cv::Point2d unit(dx / length, dy / length);
    for (double start = 0.0; start < length; start += dash_px + gap_px) {
        const double end = std::min(start + dash_px, length);
        const cv::Point p1(
            cvRound(from.x + unit.x * start),
            cvRound(from.y + unit.y * start)
        );
        const cv::Point p2(
            cvRound(from.x + unit.x * end),
            cvRound(from.y + unit.y * end)
        );
        cv::line(frame, p1, p2, color, thickness, cv::LINE_AA);
    }
}

void drawTrackingOverlay(cv::Mat& frame, const cv::Point* selectedCenter)
{
    if (frame.empty()) {
        return;
    }

    const cv::Point center(frame.cols / 2, frame.rows / 2);
    const auto overlay = AutoTracker::overlayConfig();

    const cv::Scalar center_color(255, 255, 0);
    const cv::Scalar deadzone_color(0, 210, 255);
    const cv::Scalar deadzone_outer_color = overlay.deadzoneHoldActive
        ? cv::Scalar(0, 170, 255)
        : cv::Scalar(90, 150, 210);
    const cv::Scalar line_color(0, 255, 120);

    cv::drawMarker(frame, center, center_color, cv::MARKER_CROSS, 34, 2, cv::LINE_AA);
    cv::circle(frame, center, 4, center_color, cv::FILLED, cv::LINE_AA);

    if (overlay.deadzoneEnabled) {
        const cv::Rect outer_deadzone(
            center.x - overlay.deadzoneOuterX,
            center.y - overlay.deadzoneOuterY,
            overlay.deadzoneOuterX * 2,
            overlay.deadzoneOuterY * 2
        );
        const cv::Rect deadzone(
            center.x - overlay.deadzoneX,
            center.y - overlay.deadzoneY,
            overlay.deadzoneX * 2,
            overlay.deadzoneY * 2
        );
        cv::rectangle(frame, outer_deadzone & cv::Rect(0, 0, frame.cols, frame.rows),
                      deadzone_outer_color, 1, cv::LINE_AA);
        cv::rectangle(frame, deadzone & cv::Rect(0, 0, frame.cols, frame.rows),
                      deadzone_color, 2, cv::LINE_AA);
    }

    if (selectedCenter) {
        drawDashedLine(frame, center, *selectedCenter, line_color, 2);
        cv::circle(frame, *selectedCenter, 6, line_color, cv::FILLED, cv::LINE_AA);
    }
}

double targetScore(const TargetCandidate& candidate,
                   bool havePrediction,
                   const cv::Point2d& predictedCenter)
{
    double score = static_cast<double>(candidate.confidence) *
                   app_config::kYoloCandidateConfidenceScore;
    score += std::sqrt(static_cast<double>(candidate.area)) *
             app_config::kYoloCandidateAreaScore;

    if (havePrediction) {
        const double dx = static_cast<double>(candidate.center.x) - predictedCenter.x;
        const double dy = static_cast<double>(candidate.center.y) - predictedCenter.y;
        const double dist = std::hypot(dx, dy);
        score -= std::min(dist, 1000.0) *
                 app_config::kYoloCandidateDistancePenalty;
        if (dist <= app_config::kYoloCandidateStickyRadiusPx) {
            score += app_config::kYoloCandidateStickyBonus;
        }
    }

    return score;
}

cv::Rect clampRectToFrame(const cv::Rect& rect, const cv::Size& frameSize)
{
    return rect & cv::Rect(0, 0, frameSize.width, frameSize.height);
}

cv::Rect centeredRect(const cv::Point2d& center,
                      double width,
                      double height,
                      const cv::Size& frameSize)
{
    if (frameSize.width <= 0 || frameSize.height <= 0) {
        return cv::Rect();
    }

    width = std::clamp(width, 2.0, static_cast<double>(frameSize.width));
    height = std::clamp(height, 2.0, static_cast<double>(frameSize.height));

    int w = cvRound(width);
    int h = cvRound(height);
    w = std::clamp(w, 1, frameSize.width);
    h = std::clamp(h, 1, frameSize.height);

    int x = cvRound(center.x - w * 0.5);
    int y = cvRound(center.y - h * 0.5);

    x = std::clamp(x, 0, std::max(0, frameSize.width - w));
    y = std::clamp(y, 0, std::max(0, frameSize.height - h));

    return clampRectToFrame(cv::Rect(x, y, w, h), frameSize);
}

cv::Rect centeredVirtualRect(const cv::Point2d& center,
                             double width,
                             double height)
{
    width = std::max(2.0, width);
    height = std::max(2.0, height);

    const int w = std::max(1, cvRound(width));
    const int h = std::max(1, cvRound(height));
    const int x = cvRound(center.x - w * 0.5);
    const int y = cvRound(center.y - h * 0.5);

    return cv::Rect(x, y, w, h);
}

bool sameRect(const cv::Rect& a, const cv::Rect& b)
{
    return a.x == b.x &&
           a.y == b.y &&
           a.width == b.width &&
           a.height == b.height;
}

cv::Rect dynamicRoiForTarget(const cv::Rect& targetBox,
                             const cv::Size& frameSize,
                             int lostFrames,
                             int networkInputSide,
                             double trackQuality)
{
    if (targetBox.area() <= 0 || frameSize.width <= 0 || frameSize.height <= 0) {
        return cv::Rect(0, 0, frameSize.width, frameSize.height);
    }

    const cv::Point2d center(targetBox.x + targetBox.width * 0.5,
                             targetBox.y + targetBox.height * 0.5);
    const double qualityForMin = std::max(0.01, app_config::kYoloTrackRoiQualityForMinScale);
    const double qualityRatio = clamp01(trackQuality / qualityForMin);
    const double inputScale =
        app_config::kYoloTrackRoiLowQualityInputScale +
        (app_config::kYoloTrackRoiMinInputScale -
         app_config::kYoloTrackRoiLowQualityInputScale) * qualityRatio;
    const double trackSide = std::max({
        2.0,
        static_cast<double>(networkInputSide) * inputScale,
        static_cast<double>(std::max(targetBox.width, targetBox.height)) *
            app_config::kYoloTrackRoiTargetBoxScale
    });
    if (lostFrames <= 0) {
        return centeredVirtualRect(center, trackSide, trackSide);
    }

    const double minWidth = static_cast<double>(frameSize.width) *
                            app_config::kYoloRoiMinWidthRatio;
    const double minHeight = static_cast<double>(frameSize.height) *
                             app_config::kYoloRoiMinHeightRatio;
    const double baseSide = std::max({
        trackSide,
        static_cast<double>(targetBox.width) * app_config::kYoloRoiBoxScale,
        static_cast<double>(targetBox.height) * app_config::kYoloRoiBoxScale,
        minWidth,
        minHeight
    });
    const double side = baseSide *
                        std::pow(app_config::kYoloRoiLostExpansion,
                                 std::max(0, lostFrames));

    return centeredRect(center, side, side, frameSize);
}

bool isFullFrameRoi(const cv::Rect& roi, const cv::Size& frameSize)
{
    return roi.x == 0 && roi.y == 0 &&
           roi.width == frameSize.width &&
           roi.height == frameSize.height;
}

} // namespace

void my_yolo::TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

my_yolo::my_yolo()
{
#if OPENCV_CPU_THREADS
    cv::setNumThreads(OPENCV_CPU_THREADS);
    cv::setUseOptimized(true);
#endif

    weightPath = app_config::kYoloEnginePath;

    if (weightPath.find("_1280") != std::string::npos)      yolo_width = yolo_height = 1280;
    else if (weightPath.find("_1920") != std::string::npos) yolo_width = yolo_height = 1920;
    else if (weightPath.find("_2560") != std::string::npos) yolo_width = yolo_height = 2560;
    else if (weightPath.find("_4000") != std::string::npos) yolo_width = yolo_height = 4000;
    else                                                    yolo_width = yolo_height = 640;

    CV_Assert(!weightPath.empty());
    CV_Assert(yolo_width > 0 && yolo_height > 0);

    initTensorRt();

    confThreshold = 0.3f;
    nmsThreshold  = 0.5f;

    scale  = 1.0/122.0;
    mean   = cv::Scalar(120,120,120);
    swapRB = true;

    loadClasses(app_config::kYoloClassesPath);

    classIds.reserve(256);
    confidences.reserve(256);
    boxes.reserve(256);
    keep_idx.reserve(256);

    outs.reserve(3);

    gpu_resized_u8.create(yolo_height, yolo_width, CV_8UC3);
}

my_yolo::~my_yolo()
{
    releaseTensorRt();
}

void my_yolo::releaseTensorRt()
{
    for (TrtOutputBinding& output : trtOutputs) {
        if (output.device) {
            cudaFree(output.device);
            output.device = nullptr;
        }
        output.bytes = 0;
    }
    trtOutputs.clear();

    if (trtInputDevice) {
        cudaFree(trtInputDevice);
        trtInputDevice = nullptr;
    }
    trtInputBytes = 0;

    if (trtStream) {
        cudaStreamDestroy(trtStream);
        trtStream = nullptr;
    }

    trtContext.reset();
    trtEngine.reset();
    trtRuntime.reset();
}

void my_yolo::initTensorRt()
{
    std::ifstream engineFile(weightPath, std::ios::binary | std::ios::ate);
    if (!engineFile) {
        throw std::runtime_error("[TRT] cannot open engine: " + weightPath);
    }

    const std::streamsize engineSize = engineFile.tellg();
    if (engineSize <= 0) {
        throw std::runtime_error("[TRT] empty engine: " + weightPath);
    }
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(static_cast<size_t>(engineSize));
    if (!engineFile.read(engineData.data(), engineSize)) {
        throw std::runtime_error("[TRT] cannot read engine: " + weightPath);
    }

    initLibNvInferPlugins(&trtLogger, "");

    trtRuntime.reset(nvinfer1::createInferRuntime(trtLogger));
    if (!trtRuntime) {
        throw std::runtime_error("[TRT] createInferRuntime failed");
    }

    trtEngine.reset(trtRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!trtEngine) {
        throw std::runtime_error("[TRT] deserializeCudaEngine failed: " + weightPath);
    }

    trtContext.reset(trtEngine->createExecutionContext());
    if (!trtContext) {
        throw std::runtime_error("[TRT] createExecutionContext failed");
    }

    if (cudaStreamCreateWithFlags(&trtStream, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("[TRT] cudaStreamCreateWithFlags failed");
    }

    if (!allocateTensorRtBuffers()) {
        throw std::runtime_error("[TRT] allocate buffers failed");
    }

    std::cout << "[TRT] engine loaded: " << weightPath
              << " input=" << trtInputName
              << " size=" << yolo_width << "x" << yolo_height
              << " outputs=" << trtOutputs.size()
              << std::endl;
}

bool my_yolo::allocateTensorRtBuffers()
{
    if (!trtEngine || !trtContext) {
        return false;
    }

    const int nbTensors = trtEngine->getNbIOTensors();
    trtOutputs.clear();

    for (int i = 0; i < nbTensors; ++i) {
        const char* tensorName = trtEngine->getIOTensorName(i);
        if (!tensorName) {
            continue;
        }

        const nvinfer1::TensorIOMode mode = trtEngine->getTensorIOMode(tensorName);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            trtInputName = tensorName;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            TrtOutputBinding output;
            output.name = tensorName;
            output.dataType = trtEngine->getTensorDataType(tensorName);
            trtOutputs.push_back(output);
        }
    }

    if (trtInputName.empty() || trtOutputs.empty()) {
        std::cerr << "[TRT] bad bindings: input=" << trtInputName
                  << " outputs=" << trtOutputs.size() << std::endl;
        return false;
    }

    nvinfer1::Dims inputDims = trtEngine->getTensorShape(trtInputName.c_str());
    if (inputDims.nbDims == 4) {
        inputDims.d[0] = 1;
        inputDims.d[1] = 3;
        inputDims.d[2] = yolo_height;
        inputDims.d[3] = yolo_width;
    } else if (inputDims.nbDims == 3) {
        inputDims.d[0] = 3;
        inputDims.d[1] = yolo_height;
        inputDims.d[2] = yolo_width;
    } else {
        std::cerr << "[TRT] unsupported input dims: "
                  << dimsToString(inputDims) << std::endl;
        return false;
    }

    if (!trtContext->setInputShape(trtInputName.c_str(), inputDims)) {
        std::cerr << "[TRT] setInputShape failed: "
                  << dimsToString(inputDims) << std::endl;
        return false;
    }

    const nvinfer1::DataType inputType = trtEngine->getTensorDataType(trtInputName.c_str());
    if (inputType != nvinfer1::DataType::kFLOAT) {
        std::cerr << "[TRT] unsupported input data type, expected FP32" << std::endl;
        return false;
    }

    const int64_t inputVolume = dimsVolume(inputDims);
    if (inputVolume <= 0) {
        return false;
    }
    trtInputBytes = static_cast<size_t>(inputVolume) * sizeof(float);
    if (cudaMalloc(&trtInputDevice, trtInputBytes) != cudaSuccess) {
        std::cerr << "[TRT] cudaMalloc input failed, bytes=" << trtInputBytes << std::endl;
        return false;
    }
    if (!trtContext->setTensorAddress(trtInputName.c_str(), trtInputDevice)) {
        std::cerr << "[TRT] setTensorAddress input failed" << std::endl;
        return false;
    }

    for (TrtOutputBinding& output : trtOutputs) {
        output.dims = trtContext->getTensorShape(output.name.c_str());
        const int64_t outputVolume = dimsVolume(output.dims);
        const size_t elementSize = dataTypeSize(output.dataType);
        if (outputVolume <= 0 || elementSize == 0) {
            std::cerr << "[TRT] unsupported output " << output.name
                      << " dims=" << dimsToString(output.dims) << std::endl;
            return false;
        }

        output.bytes = static_cast<size_t>(outputVolume) * elementSize;
        output.host.resize(static_cast<size_t>(outputVolume));
        if (output.dataType == nvinfer1::DataType::kHALF) {
            output.hostHalf.resize(static_cast<size_t>(outputVolume));
        }

        if (cudaMalloc(&output.device, output.bytes) != cudaSuccess) {
            std::cerr << "[TRT] cudaMalloc output failed: " << output.name
                      << " bytes=" << output.bytes << std::endl;
            return false;
        }
        if (!trtContext->setTensorAddress(output.name.c_str(), output.device)) {
            std::cerr << "[TRT] setTensorAddress output failed: "
                      << output.name << std::endl;
            return false;
        }

        std::cout << "[TRT] output " << output.name
                  << " dims=" << dimsToString(output.dims)
                  << " bytes=" << output.bytes
                  << std::endl;
    }

    return true;
}

bool my_yolo::runTensorRt(std::vector<cv::Mat>& outputMats)
{
    outputMats.clear();
    if (!trtContext || !trtStream || !trtInputDevice) {
        return false;
    }

    if (!trtContext->setTensorAddress(trtInputName.c_str(), trtInputDevice)) {
        return false;
    }
    for (TrtOutputBinding& output : trtOutputs) {
        if (!trtContext->setTensorAddress(output.name.c_str(), output.device)) {
            return false;
        }
    }

    if (!trtContext->enqueueV3(trtStream)) {
        std::cerr << "[TRT] enqueueV3 failed" << std::endl;
        return false;
    }

    for (TrtOutputBinding& output : trtOutputs) {
        if (output.dataType == nvinfer1::DataType::kFLOAT) {
            if (cudaMemcpyAsync(output.host.data(),
                                output.device,
                                output.bytes,
                                cudaMemcpyDeviceToHost,
                                trtStream) != cudaSuccess)
            {
                return false;
            }
        } else if (output.dataType == nvinfer1::DataType::kHALF) {
            if (cudaMemcpyAsync(output.hostHalf.data(),
                                output.device,
                                output.bytes,
                                cudaMemcpyDeviceToHost,
                                trtStream) != cudaSuccess)
            {
                return false;
            }
        } else {
            std::cerr << "[TRT] unsupported output data type: " << output.name << std::endl;
            return false;
        }
    }

    if (cudaStreamSynchronize(trtStream) != cudaSuccess) {
        std::cerr << "[TRT] stream synchronize failed" << std::endl;
        return false;
    }

    for (TrtOutputBinding& output : trtOutputs) {
        if (output.dataType == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < output.host.size(); ++i) {
                output.host[i] = __half2float(output.hostHalf[i]);
            }
        }

        std::vector<int> sizes;
        sizes.reserve(output.dims.nbDims);
        for (int i = 0; i < output.dims.nbDims; ++i) {
            sizes.push_back(static_cast<int>(output.dims.d[i]));
        }

        outputMats.emplace_back(static_cast<int>(sizes.size()),
                                sizes.data(),
                                CV_32F,
                                output.host.data());
    }

    return true;
}

void my_yolo::loadClasses(const std::string& namesPath)
{
    classes.clear();
    std::ifstream ifs(namesPath);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) classes.push_back(line);
    }
}

yolo_output my_yolo::run(cv::Mat &frame, double captureMs)
{
    // PERF timers
    cv::TickMeter tm_total, tm_blob, tm_fwd, tm_decode, tm_nms, tm_scale, tm_drawpack;

    tm_total.start();

    output.points.clear();
    output.area.clear();

    classIds.clear();
    confidences.clear();
    boxes.clear();
    keep_idx.clear();

    CV_Assert(!frame.empty());
    CV_Assert(frame.type() == CV_8UC3);

    const uint64_t frameNowMs = monotonicMs();
    auto predictTrackCenter = [this](uint64_t nowMsValue) {
        if (!have_track_model || track_last_update_ms == 0) {
            return track_center_f;
        }

        const double rawDtMs = nowMsValue >= track_last_update_ms
            ? static_cast<double>(nowMsValue - track_last_update_ms)
            : 0.0;
        const double dtMs = std::clamp(rawDtMs,
                                       0.0,
                                       app_config::kYoloMotionMaxPredictMs);
        const double dtSec = dtMs / 1000.0;
        return cv::Point2d(track_center_f.x + track_velocity_px_s.x * dtSec,
                           track_center_f.y + track_velocity_px_s.y * dtSec);
    };

    const cv::Rect fullFrameRoi(0, 0, frame.cols, frame.rows);
    const cv::Point2d predictedCenter = have_track_model
        ? predictTrackCenter(frameNowMs)
        : cv::Point2d(last_target_center.x, last_target_center.y);
    const cv::Size2d predictedBoxSize = have_track_model
        ? track_box_size_f
        : cv::Size2d(last_target_box.width, last_target_box.height);
    const cv::Rect roiAnchorBox = have_track_model &&
                                  predictedBoxSize.width > 0.0 &&
                                  predictedBoxSize.height > 0.0
        ? rectFromCenterSize(predictedCenter, predictedBoxSize)
        : last_target_box;
    cv::Rect inferenceRoi = fullFrameRoi;
    cv::Rect visibleInferenceRoi = fullFrameRoi;
    const bool periodicFullScan =
        app_config::kYoloRoiFullScanPeriodFrames > 0 &&
        roi_frames_since_full_scan >= app_config::kYoloRoiFullScanPeriodFrames;
    const bool canUseDynamicRoi =
        app_config::kYoloDynamicRoiEnabled &&
        have_last_target &&
        last_target_box.area() > 0 &&
        !periodicFullScan &&
        lost_target_frames <= app_config::kYoloRoiMaxLostFramesBeforeSearch;

    if (canUseDynamicRoi) {
        const int networkInputSide = std::max(yolo_width, yolo_height);
        inferenceRoi = dynamicRoiForTarget(roiAnchorBox,
                                           frame.size(),
                                           lost_target_frames,
                                           networkInputSide,
                                           track_quality);
        if (inferenceRoi.area() <= 0) {
            inferenceRoi = fullFrameRoi;
        }
    }

    if (inferenceRoi.area() <= 0) {
        inferenceRoi = fullFrameRoi;
    }
    visibleInferenceRoi = clampRectToFrame(inferenceRoi, frame.size());
    if (visibleInferenceRoi.area() <= 0) {
        inferenceRoi = fullFrameRoi;
        visibleInferenceRoi = fullFrameRoi;
    }

    const bool roiMode = !isFullFrameRoi(inferenceRoi, frame.size());
    const bool paddedRoi = roiMode && !sameRect(inferenceRoi, visibleInferenceRoi);
    const char* pipelineModeBase = roiMode
        ? (lost_target_frames > 0 ? "LOST_ROI" : "TRACK_ROI")
        : "SEARCH_FULL";
    const std::string pipelineMode = paddedRoi
        ? std::string(pipelineModeBase) + "_PAD"
        : std::string(pipelineModeBase);

    cv::Mat inferenceFrame;
    if (!roiMode) {
        inferenceFrame = frame;
    } else if (paddedRoi) {
        padded_inference_frame.create(inferenceRoi.height, inferenceRoi.width, frame.type());
        // Raw mean becomes zero after the model normalization, so padding is neutral.
        padded_inference_frame.setTo(mean);

        const cv::Rect dstRoi(
            visibleInferenceRoi.x - inferenceRoi.x,
            visibleInferenceRoi.y - inferenceRoi.y,
            visibleInferenceRoi.width,
            visibleInferenceRoi.height
        );
        frame(visibleInferenceRoi).copyTo(padded_inference_frame(dstRoi));
        inferenceFrame = padded_inference_frame;
    } else {
        inferenceFrame = frame(visibleInferenceRoi);
    }

    // ============================
    // 2.1 blob (preprocess)
    // ============================
    tm_blob.start();
    bool inputReady = false;
#if ENABLE_CUDA_PREPROCESS
    try {
        gpu_frame_u8.upload(inferenceFrame, gpu_stream);

        const bool inputAlreadySized =
            inferenceFrame.cols == yolo_width &&
            inferenceFrame.rows == yolo_height;
        cv::cuda::GpuMat* preprocessInput = &gpu_frame_u8;
        if (!inputAlreadySized) {
            cv::cuda::resize(gpu_frame_u8, gpu_resized_u8,
                             cv::Size(yolo_width, yolo_height),
                             0, 0, cv::INTER_LINEAR, gpu_stream);
            preprocessInput = &gpu_resized_u8;
        }

        cudaStream_t s = cv::cuda::StreamAccessor::getStream(gpu_stream);

        bool ok = cuda_preprocess_bgr_to_nchw_ptr(
            *preprocessInput,
            static_cast<float*>(trtInputDevice),
            yolo_width, yolo_height,
            (float)scale,
            (float)mean[0], (float)mean[1], (float)mean[2],
            swapRB,
            s
        );

        gpu_stream.waitForCompletion();
        if (!ok) throw std::runtime_error("cuda_preprocess_bgr_to_nchw failed");
        inputReady = true;
    } catch (...) {
        cv::Mat blob = cv::dnn::blobFromImage(inferenceFrame, scale,
                                              cv::Size(yolo_width, yolo_height),
                                              mean, swapRB, false, CV_32F);
        inputReady = (cudaMemcpyAsync(trtInputDevice,
                                      blob.ptr<float>(),
                                      trtInputBytes,
                                      cudaMemcpyHostToDevice,
                                      trtStream) == cudaSuccess &&
                      cudaStreamSynchronize(trtStream) == cudaSuccess);
    }
#else
    cv::Mat blob = cv::dnn::blobFromImage(inferenceFrame, scale,
                                          cv::Size(yolo_width, yolo_height),
                                          mean, swapRB, false, CV_32F);
    inputReady = (cudaMemcpyAsync(trtInputDevice,
                                  blob.ptr<float>(),
                                  trtInputBytes,
                                  cudaMemcpyHostToDevice,
                                  trtStream) == cudaSuccess &&
                  cudaStreamSynchronize(trtStream) == cudaSuccess);
#endif
    if (!inputReady) {
        throw std::runtime_error("[TRT] input preprocess/copy failed");
    }
    tm_blob.stop();

    // ============================
    // 2.2 forward (inference)
    // ============================
    tm_fwd.start();
    outs.clear();
    if (!runTensorRt(outs)) {
        throw std::runtime_error("[TRT] inference failed");
    }
    tm_fwd.stop();

    // ============================
    // 2.3 decode (parse outputs -> boxes)
    // ============================
    tm_decode.start();
    for (cv::Mat& preds : outs)
    {
        if (preds.dims > 2) {
            preds = preds.reshape(1, preds.size[preds.dims - 2]);
        } else if (preds.rows == 1) {
            preds = preds.reshape(1, preds.cols);
        }

        const int rows = preds.rows;
        const int cols = preds.cols;

        for (int i = 0; i < rows; ++i)
        {
            const float* det = preds.ptr<float>(i);
            float obj_conf = det[4];
            if (obj_conf < confThreshold) continue;

            int bestClass = -1;
            float bestScore = 0.f;
            for (int c = 5; c < cols; ++c) {
                float s = det[c];
                if (s > bestScore) { bestScore = s; bestClass = (c - 5); }
            }

            float conf = bestScore * obj_conf;
            if (conf < confThreshold) continue;

            float cx = det[0];
            float cy = det[1];
            float w  = det[2];
            float h  = det[3];

            float x = cx - 0.5f * w;
            float y = cy - 0.5f * h;

            boxes.emplace_back((int)x, (int)y, (int)w, (int)h);
            classIds.emplace_back(bestClass);
            confidences.emplace_back(conf);
        }
    }
    tm_decode.stop();

    // ============================
    // 2.4 nms
    // ============================
    tm_nms.start();
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, keep_idx);
    tm_nms.stop();

    // ============================
    // 2.5 scale (map to original frame)
    // ============================
    tm_scale.start();
    const float sx = (float)inferenceRoi.width / (float)yolo_width;
    const float sy = (float)inferenceRoi.height / (float)yolo_height;
    tm_scale.stop();

    // ============================
    // 2.6 draw+pack
    // ============================
    tm_drawpack.start();
    int kept = 0;
    std::vector<TargetCandidate> candidates;
    candidates.reserve(keep_idx.size());

    const bool jumpGuardActive = have_track_model &&
                                 (track_confirmed || acquire_hit_frames > 0) &&
                                 predictedBoxSize.width > 0.0 &&
                                 predictedBoxSize.height > 0.0;
    const double predictDtMs = have_track_model && track_last_update_ms != 0 &&
                               frameNowMs >= track_last_update_ms
        ? static_cast<double>(frameNowMs - track_last_update_ms)
        : 0.0;
    const double predictDtSec = std::clamp(
        predictDtMs / 1000.0,
        0.02,
        app_config::kYoloMotionMaxPredictMs / 1000.0
    );
    const double predictedDiag = std::max(1.0, hypotSize(predictedBoxSize));
    const double predictedSpeed = std::hypot(track_velocity_px_s.x,
                                             track_velocity_px_s.y);

    auto isSuspiciousCandidate = [&](const TargetCandidate& candidate) {
        if (!jumpGuardActive) {
            return false;
        }

        const double dx = static_cast<double>(candidate.center.x) - predictedCenter.x;
        const double dy = static_cast<double>(candidate.center.y) - predictedCenter.y;
        const double dist = std::hypot(dx, dy);
        double allowedDist =
            app_config::kYoloTrackMaxJumpBasePx +
            predictedDiag * app_config::kYoloTrackMaxJumpBoxDiagRatio +
            predictedSpeed * predictDtSec * app_config::kYoloTrackVelocityGateScale;
        allowedDist *= 1.0 +
            (1.0 - clamp01(track_quality)) * app_config::kYoloTrackLowQualityGateRelax;
        allowedDist += static_cast<double>(lost_target_frames) *
                       app_config::kYoloTrackMaxJumpBasePx * 0.5;

        if (dist > allowedDist) {
            return true;
        }

        const double ratioW = std::max(
            static_cast<double>(candidate.box.width) / std::max(1.0, predictedBoxSize.width),
            predictedBoxSize.width / std::max(1.0, static_cast<double>(candidate.box.width))
        );
        const double ratioH = std::max(
            static_cast<double>(candidate.box.height) / std::max(1.0, predictedBoxSize.height),
            predictedBoxSize.height / std::max(1.0, static_cast<double>(candidate.box.height))
        );
        const double maxRatio =
            app_config::kYoloTrackMaxBoxSizeChangeRatio *
            (1.0 + 0.20 * static_cast<double>(lost_target_frames));

        return ratioW > maxRatio || ratioH > maxRatio;
    };

    for (int idx : keep_idx)
    {
        cv::Rect b = boxes[idx];
        b.x = inferenceRoi.x + (int)std::round(b.x * sx);
        b.y = inferenceRoi.y + (int)std::round(b.y * sy);
        b.width  = (int)std::round(b.width  * sx);
        b.height = (int)std::round(b.height * sy);

        b &= cv::Rect(0,0,frame.cols,frame.rows);
        if (b.area() <= 0) continue;

        TargetCandidate candidate;
        candidate.index = idx;
        candidate.box = b;
        candidate.center = cv::Point(b.x + b.width / 2, b.y + b.height / 2);
        candidate.area = b.area();
        candidate.confidence = confidences[idx];
        candidate.score = targetScore(candidate, have_track_model, predictedCenter);
        candidate.suspicious = isSuspiciousCandidate(candidate);

        candidates.push_back(candidate);
        kept++;
    }

    int selected = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int suspiciousSelected = -1;
    double bestSuspiciousScore = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i].suspicious) {
            if (candidates[i].score > bestSuspiciousScore) {
                bestSuspiciousScore = candidates[i].score;
                suspiciousSelected = i;
            }
            continue;
        }

        if (candidates[i].score > best_score) {
            best_score = candidates[i].score;
            selected = i;
        }
    }

    bool suspiciousOnly = false;
    if (selected < 0 && suspiciousSelected >= 0) {
        suspiciousOnly = true;
        ++suspicious_frames;
        track_quality = std::max(0.0,
                                 track_quality - app_config::kYoloTrackQualitySuspiciousDecay);

        if (suspicious_frames >= app_config::kYoloTrackSuspiciousResetFrames) {
            selected = suspiciousSelected;
            have_track_model = false;
            track_confirmed = false;
            acquire_hit_frames = 0;
            suspicious_frames = 0;
            track_quality = 0.0;
        }
    } else if (selected >= 0) {
        suspicious_frames = 0;
    }

    const cv::Point* selectedCenter = nullptr;
    bool coastedThisFrame = false;
    cv::Point coastCenterForOverlay;
    cv::Rect coastBoxForOverlay;
    if (selected >= 0) {
        selectedCenter = &candidates[selected].center;
    }

    if (app_config::kYoloDynamicRoiEnabled &&
        app_config::kYoloDynamicRoiDrawOverlay &&
        roiMode)
    {
        const cv::Scalar roiColor = lost_target_frames > 0
            ? cv::Scalar(0, 170, 255)
            : cv::Scalar(255, 180, 0);
        cv::rectangle(frame, visibleInferenceRoi, roiColor, 2, cv::LINE_AA);
        cv::putText(frame,
                    pipelineMode,
                    cv::Point(visibleInferenceRoi.x + 8,
                              std::max(24, visibleInferenceRoi.y + 28)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    roiColor,
                    2,
                    cv::LINE_AA);
    }

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
        const TargetCandidate& candidate = candidates[i];
        const bool is_selected = (i == selected);
        const cv::Scalar color = is_selected ? cv::Scalar(0, 255, 0)
                                             : cv::Scalar(0, 0, 255);

        cv::rectangle(frame, candidate.box, color, is_selected ? 3 : 2);

        std::string label = cv::format("%.2f", candidate.confidence);
        if (!classes.empty() &&
            classIds[candidate.index] >= 0 &&
            classIds[candidate.index] < static_cast<int>(classes.size()))
        {
            label = classes[classIds[candidate.index]] + ":" + label;
        }
        cv::putText(frame, label, cv::Point(candidate.box.x, std::max(0, candidate.box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

        output.points.push_back(candidate.center);
        output.area.push_back(candidate.area);
    }

    tm_drawpack.stop();
    const double postprocessMs =
        ms(tm_decode) + ms(tm_nms) + ms(tm_scale) + ms(tm_drawpack);
    const latency_monitor::Token latencyToken = latency_monitor::beginCameraFrame(
        captureMs,
        ms(tm_blob),
        ms(tm_fwd),
        postprocessMs);

    auto resetTrackModel = [this]() {
        have_last_target = false;
        have_track_model = false;
        track_confirmed = false;
        last_target_center = cv::Point();
        last_target_box = cv::Rect();
        track_center_f = cv::Point2d();
        track_velocity_px_s = cv::Point2d();
        track_box_size_f = cv::Size2d();
        track_last_update_ms = 0;
        track_last_detection_ms = 0;
        track_quality = 0.0;
        acquire_hit_frames = 0;
        suspicious_frames = 0;
        lost_target_frames = 0;
    };

    auto filteredTrackBox = [this, &frame]() {
        return rectFromCenterSize(track_center_f, track_box_size_f) &
               cv::Rect(0, 0, frame.cols, frame.rows);
    };

    auto updateTrackModel = [&](const TargetCandidate& target) {
        const cv::Point2d measurementCenter(target.center.x, target.center.y);
        const cv::Size2d measurementSize(target.box.width, target.box.height);

        if (!have_track_model ||
            track_last_update_ms == 0 ||
            track_box_size_f.width <= 0.0 ||
            track_box_size_f.height <= 0.0)
        {
            track_center_f = measurementCenter;
            track_velocity_px_s = cv::Point2d();
            track_box_size_f = measurementSize;
            have_track_model = true;
        } else {
            const double rawDtMs = frameNowMs >= track_last_update_ms
                ? static_cast<double>(frameNowMs - track_last_update_ms)
                : 0.0;
            const double dtSec = std::clamp(rawDtMs / 1000.0,
                                            0.001,
                                            app_config::kYoloMotionMaxPredictMs / 1000.0);
            const cv::Point2d predicted(
                track_center_f.x + track_velocity_px_s.x * dtSec,
                track_center_f.y + track_velocity_px_s.y * dtSec
            );
            const cv::Point2d residual(measurementCenter.x - predicted.x,
                                       measurementCenter.y - predicted.y);

            track_center_f = cv::Point2d(
                predicted.x + app_config::kYoloMotionAlpha * residual.x,
                predicted.y + app_config::kYoloMotionAlpha * residual.y
            );
            track_velocity_px_s = cv::Point2d(
                track_velocity_px_s.x +
                    app_config::kYoloMotionBeta * residual.x / dtSec,
                track_velocity_px_s.y +
                    app_config::kYoloMotionBeta * residual.y / dtSec
            );

            const double speed = std::hypot(track_velocity_px_s.x,
                                            track_velocity_px_s.y);
            if (speed > app_config::kYoloMotionMaxVelocityPxPerSec &&
                speed > 1.0)
            {
                const double scale =
                    app_config::kYoloMotionMaxVelocityPxPerSec / speed;
                track_velocity_px_s.x *= scale;
                track_velocity_px_s.y *= scale;
            }

            const double boxAlpha = clamp01(app_config::kYoloTrackBoxLpfAlpha);
            track_box_size_f = cv::Size2d(
                (1.0 - boxAlpha) * track_box_size_f.width +
                    boxAlpha * measurementSize.width,
                (1.0 - boxAlpha) * track_box_size_f.height +
                    boxAlpha * measurementSize.height
            );
        }

        track_last_update_ms = frameNowMs;
        track_last_detection_ms = frameNowMs;
        track_quality = std::min(
            1.0,
            track_quality +
                app_config::kYoloTrackQualityHitGain *
                    (0.5 + 0.5 * static_cast<double>(target.confidence))
        );
        ++acquire_hit_frames;
        lost_target_frames = 0;

        if (!track_confirmed &&
            acquire_hit_frames >= app_config::kYoloTrackAcquireFrames &&
            track_quality >= app_config::kYoloTrackAcquireMinQuality)
        {
            track_confirmed = true;
        }

        last_target_center = roundPoint(track_center_f);
        last_target_box = filteredTrackBox();
        have_last_target = last_target_box.area() > 0;
    };

    if (selected >= 0) {
        const TargetCandidate& target = candidates[selected];
        updateTrackModel(target);

        if (track_confirmed && have_last_target) {
            TargetManager::submitCameraTarget(last_target_center,
                                              last_target_box,
                                              frame.size());
        } else {
            TargetManager::submitCameraMiss();
        }

        if (roiMode) {
            ++roi_frames_since_full_scan;
        } else {
            roi_frames_since_full_scan = 0;
        }
    } else {
        bool canCoastThisFrame = false;
        if (have_track_model) {
            ++lost_target_frames;
            track_center_f = predictTrackCenter(frameNowMs);
            track_last_update_ms = frameNowMs;
            acquire_hit_frames = 0;
            const double sinceDetectionMs =
                track_last_detection_ms != 0 && frameNowMs >= track_last_detection_ms
                    ? static_cast<double>(frameNowMs - track_last_detection_ms)
                    : 0.0;
            const bool withinCoastWindow =
                app_config::kYoloTrackCoastEnabled &&
                track_confirmed &&
                track_last_detection_ms != 0 &&
                lost_target_frames <= app_config::kYoloTrackCoastMaxFrames &&
                sinceDetectionMs <= app_config::kYoloTrackCoastMaxMs;

            if (!suspiciousOnly) {
                const double missDecay = withinCoastWindow
                    ? app_config::kYoloTrackCoastQualityDecay
                    : app_config::kYoloTrackQualityMissDecay;
                track_quality = std::max(
                    0.0,
                    track_quality - missDecay
                );
            }

            last_target_center = roundPoint(track_center_f);
            last_target_box = filteredTrackBox();
            have_last_target = last_target_box.area() > 0;

            canCoastThisFrame =
                withinCoastWindow &&
                have_last_target &&
                track_quality >= app_config::kYoloTrackCoastMinQuality;
        }

        if (have_track_model &&
            (lost_target_frames > app_config::kYoloTrackMemoryFrames ||
             track_quality < app_config::kYoloTrackReleaseQuality))
        {
            resetTrackModel();
            roi_frames_since_full_scan = 0;
            canCoastThisFrame = false;
        } else if (roiMode) {
            ++roi_frames_since_full_scan;
        } else {
            roi_frames_since_full_scan = 0;
        }

        if (canCoastThisFrame && have_last_target) {
            coastCenterForOverlay = clampPointToFrame(last_target_center, frame.size());
            coastBoxForOverlay = last_target_box;
            selectedCenter = &coastCenterForOverlay;
            coastedThisFrame = true;

            TargetManager::submitCameraPredictedTarget(coastCenterForOverlay,
                                                       coastBoxForOverlay,
                                                       frame.size());
        } else {
            TargetManager::submitCameraMiss();
        }
    }
    last_inference_roi = inferenceRoi;
    latency_monitor::finishFrameWithoutSend(latencyToken);
    latency_monitor::clearCurrentCameraToken();

    if (coastedThisFrame && coastBoxForOverlay.area() > 0) {
        const cv::Scalar coastColor(0, 220, 255);
        cv::rectangle(frame, coastBoxForOverlay, coastColor, 2, cv::LINE_AA);
        cv::putText(frame,
                    "COAST",
                    cv::Point(coastBoxForOverlay.x,
                              std::max(0, coastBoxForOverlay.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    coastColor,
                    2,
                    cv::LINE_AA);
    }

    drawTrackingOverlay(frame, selectedCenter);

    tm_total.stop();

    // ============================
    // PRINT PERF (каждые N вызовов run())
    // но делим на perf_divisor, чтобы это было “на кадр” при пропуске инференса
    // ============================
    frame_counter++;
    if (frame_counter % perf_print_every == 0)
    {
        const double div = (double)perf_divisor;

        if (app_config::kYoloDynamicRoiEnabled) {
            std::cerr << "[YOLO PIPE] "
                      << "mode=" << pipelineMode
                      << " roi=" << inferenceRoi.x << "," << inferenceRoi.y
                      << " " << inferenceRoi.width << "x" << inferenceRoi.height
                      << " visible=" << visibleInferenceRoi.x << "," << visibleInferenceRoi.y
                      << " " << visibleInferenceRoi.width << "x" << visibleInferenceRoi.height
                      << " pad=" << (paddedRoi ? 1 : 0)
                      << " q=" << track_quality
                      << " acq=" << acquire_hit_frames
                      << " confirmed=" << (track_confirmed ? 1 : 0)
                      << " suspicious=" << suspicious_frames
                      << " lost=" << lost_target_frames
                      << " coast=" << (coastedThisFrame ? 1 : 0)
                      << " kept=" << kept
                      << " selected=" << (selected >= 0 ? 1 : 0)
                      << std::endl;
        }

//        std::cerr
//            << "[PERF YOLO] "
//            << "2.1.blob=" << (ms(tm_blob) / div) << "ms, "
//            << "2.2.fwd=" << (ms(tm_fwd) / div) << "ms, "
//            << "2.3.decode=" << (ms(tm_decode) / div) << "ms, "
//            << "2.4.nms=" << (ms(tm_nms) / div) << "ms, "
//            << "2.5.scale=" << (ms(tm_scale) / div) << "ms, "
//            << "2.6.draw+pack=" << (ms(tm_drawpack) / div) << "ms, "
//            << "2.total=" << (ms(tm_total) / div) << "ms, "
//            << "kept=" << kept
//            << "\n";
    }

    return output;
}
