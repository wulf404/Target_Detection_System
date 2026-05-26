#include "deepstream_yolo.h"

#include "app_config.h"
#include "auto_tracker.h"
#include "target_manager.h"

#include <cuda_fp16.h>
#include <gst/video/video.h>

#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsinfer.h>
#include <nvdsmeta.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int TARGET_MEMORY_FRAMES = 15;
constexpr double CONF_SCORE = 1000.0;
constexpr double AREA_SCORE = 0.0006;
constexpr double DIST_PENALTY = 0.6;
constexpr double STICKY_RADIUS_PX = 260.0;
constexpr double STICKY_BONUS = 180.0;

void initGstOnce()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        int argc = 0;
        char** argv = nullptr;
        gst_init(&argc, &argv);
    });
}

int dimsElements(const NvDsInferDims& dims)
{
    if (dims.numElements > 0) {
        return static_cast<int>(dims.numElements);
    }

    int total = 1;
    for (unsigned int i = 0; i < dims.numDims; ++i) {
        total *= static_cast<int>(dims.d[i]);
    }
    return total;
}

bool inferRowsCols(const NvDsInferDims& dims, int& rows, int& cols)
{
    if (dims.numDims == 0) {
        return false;
    }

    const int total = dimsElements(dims);
    cols = static_cast<int>(dims.d[dims.numDims - 1]);
    if (cols < 6 || total <= 0 || total % cols != 0) {
        return false;
    }

    rows = total / cols;
    return rows > 0;
}

float tensorValue(const NvDsInferLayerInfo& layer, const void* data, int index)
{
    if (!data) {
        return 0.0f;
    }

    switch (layer.dataType) {
    case FLOAT:
        return static_cast<const float*>(data)[index];
    case HALF:
        return __half2float(static_cast<const __half*>(data)[index]);
    case INT8:
        return static_cast<float>(static_cast<const std::int8_t*>(data)[index]);
    case INT32:
        return static_cast<float>(static_cast<const std::int32_t*>(data)[index]);
    case INT64:
        return static_cast<float>(static_cast<const std::int64_t*>(data)[index]);
    default:
        return 0.0f;
    }
}

void drawDashedLine(cv::Mat& frame,
                    const cv::Point& from,
                    const cv::Point& to,
                    const cv::Scalar& color,
                    int thickness = 2,
                    double dashPx = 18.0,
                    double gapPx = 10.0)
{
    const double dx = static_cast<double>(to.x - from.x);
    const double dy = static_cast<double>(to.y - from.y);
    const double length = std::hypot(dx, dy);
    if (length < 1.0) {
        return;
    }

    const cv::Point2d unit(dx / length, dy / length);
    for (double start = 0.0; start < length; start += dashPx + gapPx) {
        const double end = std::min(start + dashPx, length);
        const cv::Point p1(cvRound(from.x + unit.x * start),
                           cvRound(from.y + unit.y * start));
        const cv::Point p2(cvRound(from.x + unit.x * end),
                           cvRound(from.y + unit.y * end));
        cv::line(frame, p1, p2, color, thickness, cv::LINE_AA);
    }
}

cv::Point scalePointToDisplay(const cv::Point& point,
                              const cv::Size& sourceSize,
                              const cv::Size& displaySize)
{
    if (sourceSize.width <= 0 || sourceSize.height <= 0) {
        return point;
    }

    const double sx = static_cast<double>(displaySize.width) /
                      static_cast<double>(sourceSize.width);
    const double sy = static_cast<double>(displaySize.height) /
                      static_cast<double>(sourceSize.height);
    return cv::Point(cvRound(static_cast<double>(point.x) * sx),
                     cvRound(static_cast<double>(point.y) * sy));
}

cv::Rect scaleRectToDisplay(const cv::Rect& rect,
                            const cv::Size& sourceSize,
                            const cv::Size& displaySize)
{
    if (sourceSize.width <= 0 || sourceSize.height <= 0) {
        return rect & cv::Rect(0, 0, displaySize.width, displaySize.height);
    }

    const double sx = static_cast<double>(displaySize.width) /
                      static_cast<double>(sourceSize.width);
    const double sy = static_cast<double>(displaySize.height) /
                      static_cast<double>(sourceSize.height);
    cv::Rect scaled(cvRound(static_cast<double>(rect.x) * sx),
                    cvRound(static_cast<double>(rect.y) * sy),
                    cvRound(static_cast<double>(rect.width) * sx),
                    cvRound(static_cast<double>(rect.height) * sy));
    return scaled & cv::Rect(0, 0, displaySize.width, displaySize.height);
}

void drawTrackingOverlay(cv::Mat& frame,
                         const cv::Point* selectedCenter,
                         const cv::Size& sourceSize)
{
    if (frame.empty()) {
        return;
    }

    const cv::Point center(frame.cols / 2, frame.rows / 2);
    const cv::Size displaySize(frame.cols, frame.rows);
    const auto overlay = AutoTracker::overlayConfig();
    const double sx = sourceSize.width > 0
        ? static_cast<double>(displaySize.width) / static_cast<double>(sourceSize.width)
        : 1.0;
    const double sy = sourceSize.height > 0
        ? static_cast<double>(displaySize.height) / static_cast<double>(sourceSize.height)
        : 1.0;
    const int deadzoneX = std::max(1, cvRound(static_cast<double>(overlay.deadzoneX) * sx));
    const int deadzoneY = std::max(1, cvRound(static_cast<double>(overlay.deadzoneY) * sy));
    const int deadzoneOuterX = std::max(1, cvRound(static_cast<double>(overlay.deadzoneOuterX) * sx));
    const int deadzoneOuterY = std::max(1, cvRound(static_cast<double>(overlay.deadzoneOuterY) * sy));

    const cv::Scalar centerColor(255, 255, 0);
    const cv::Scalar deadzoneColor(0, 210, 255);
    const cv::Scalar deadzoneOuterColor = overlay.deadzoneHoldActive
        ? cv::Scalar(0, 170, 255)
        : cv::Scalar(90, 150, 210);
    const cv::Scalar lineColor(0, 255, 120);

    cv::drawMarker(frame, center, centerColor, cv::MARKER_CROSS, 34, 2, cv::LINE_AA);
    cv::circle(frame, center, 4, centerColor, cv::FILLED, cv::LINE_AA);

    if (overlay.deadzoneEnabled) {
        const cv::Rect outerDeadzone(center.x - deadzoneOuterX,
                                     center.y - deadzoneOuterY,
                                     deadzoneOuterX * 2,
                                     deadzoneOuterY * 2);
        const cv::Rect deadzone(center.x - deadzoneX,
                                center.y - deadzoneY,
                                deadzoneX * 2,
                                deadzoneY * 2);
        cv::rectangle(frame, outerDeadzone & cv::Rect(0, 0, frame.cols, frame.rows),
                      deadzoneOuterColor, 1, cv::LINE_AA);
        cv::rectangle(frame, deadzone & cv::Rect(0, 0, frame.cols, frame.rows),
                      deadzoneColor, 2, cv::LINE_AA);
    }

    if (selectedCenter) {
        drawDashedLine(frame, center, *selectedCenter, lineColor, 2);
        cv::circle(frame, *selectedCenter, 6, lineColor, cv::FILLED, cv::LINE_AA);
    }
}

double targetScore(const DeepStreamYolo::Detection& detection,
                   bool haveLastTarget,
                   const cv::Point& lastTargetCenter)
{
    double score = static_cast<double>(detection.confidence) * CONF_SCORE +
                   static_cast<double>(detection.area) * AREA_SCORE;

    if (haveLastTarget) {
        const double dx = static_cast<double>(detection.center.x - lastTargetCenter.x);
        const double dy = static_cast<double>(detection.center.y - lastTargetCenter.y);
        const double dist = std::hypot(dx, dy);
        score -= dist * DIST_PENALTY;
        if (dist < STICKY_RADIUS_PX) {
            score += STICKY_BONUS;
        }
    }
    return score;
}

} // namespace

DeepStreamYolo::DeepStreamYolo()
{
    initGstOnce();
    loadClasses();
}

DeepStreamYolo::~DeepStreamYolo()
{
    close();
}

bool DeepStreamYolo::writeInferConfig()
{
    std::ifstream onnxFile(app_config::kDeepStreamOnnxPath, std::ios::binary);
    std::ifstream engineFile(app_config::kDeepStreamEnginePath, std::ios::binary);
    if (!engineFile && !onnxFile) {
        pipelineErrorText = std::string("neither engine nor ONNX exists: engine=") +
            app_config::kDeepStreamEnginePath + " onnx=" + app_config::kDeepStreamOnnxPath;
        fatalError = true;
        std::cerr << "[DS][CONFIG][FATAL] " << pipelineErrorText << std::endl;
        return false;
    }

    waitingForEngineBuild = !engineFile;
    std::cout << "[DS][CONFIG] engine=" << app_config::kDeepStreamEnginePath
              << " input=" << app_config::kDeepStreamNetworkInputWidth
              << "x" << app_config::kDeepStreamNetworkInputHeight;
    if (!engineFile) {
        std::cout << " engine-missing: nvinfer will build it from ONNX="
                  << app_config::kDeepStreamOnnxPath;
    } else {
        std::cout << " engine-present";
    }
    std::cout << std::endl;

    std::ofstream file(app_config::kDeepStreamInferConfigPath);
    if (!file) {
        pipelineErrorText = std::string("cannot write infer config: ") +
            app_config::kDeepStreamInferConfigPath;
        fatalError = true;
        std::cerr << "[DS][CONFIG][FATAL] " << pipelineErrorText << std::endl;
        return false;
    }

    file << "[property]\n"
         << "gpu-id=0\n"
         << "onnx-file=" << app_config::kDeepStreamOnnxPath << "\n"
         << "model-engine-file=" << app_config::kDeepStreamEnginePath << "\n"
         << "labelfile-path=" << app_config::kYoloClassesPath << "\n"
         << "batch-size=1\n"
         << "process-mode=1\n"
         << "network-type=100\n"
         << "network-mode=2\n"
         << "force-implicit-batch-dim=0\n"
         << "gie-unique-id=1\n"
         << "infer-dims=3;" << app_config::kDeepStreamNetworkInputHeight
         << ";" << app_config::kDeepStreamNetworkInputWidth << "\n"
         << "maintain-aspect-ratio=0\n"
         << "symmetric-padding=0\n"
         << "model-color-format=0\n"
         << "net-scale-factor=" << (1.0 / 122.0) << "\n"
         << "offsets=120;120;120\n"
         << "output-tensor-meta=1\n";

    return true;
}

std::vector<DeepStreamYolo::PipelineDefinition>
DeepStreamYolo::buildPipelineCandidates(const std::string& devicePath,
                                        int width,
                                        int height,
                                        int fps) const
{
    std::vector<PipelineDefinition> candidates;
    const int muxWidth = app_config::kDeepStreamScaleInMuxToNetworkInput
        ? app_config::kDeepStreamNetworkInputWidth
        : width;
    const int muxHeight = app_config::kDeepStreamScaleInMuxToNetworkInput
        ? app_config::kDeepStreamNetworkInputHeight
        : height;

    std::ostringstream inferTail;
    inferTail
        << "queue leaky=2 max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
        << "mux.sink_0 nvstreammux name=mux batch-size=1 width=" << muxWidth
        << " height=" << muxHeight
        << " enable-padding=0 live-source=1 batched-push-timeout=10000 ! "
        << "nvinfer name=primary config-file-path=" << app_config::kDeepStreamInferConfigPath << " ! "
        << "queue leaky=2 max-size-buffers=1 ! "
        << "nvvideoconvert ! "
        << "video/x-raw(memory:NVMM),format=NV12,width=" << app_config::kDeepStreamDisplayWidth
        << ",height=" << app_config::kDeepStreamDisplayHeight << " ! "
        << "nvvideoconvert ! "
        << "video/x-raw,format=BGRx,width=" << app_config::kDeepStreamDisplayWidth
        << ",height=" << app_config::kDeepStreamDisplayHeight << " ! "
        << "videoconvert ! "
        << "video/x-raw,format=BGR,width=" << app_config::kDeepStreamDisplayWidth
        << ",height=" << app_config::kDeepStreamDisplayHeight << " ! "
        << "appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true";

    {
        std::ostringstream text;
        text << "v4l2src device=" << devicePath << " io-mode=2 do-timestamp=true ! "
             << "image/jpeg,width=" << width << ",height=" << height
             << ",framerate=" << fps << "/1 ! "
             << "jpegparse ! "
             << "nvv4l2decoder mjpeg=1 ! "
             << inferTail.str();
        candidates.push_back({"mjpeg-nvdec-nv12", text.str()});
    }

    {
        std::ostringstream text;
        text << "v4l2src device=" << devicePath << " io-mode=2 do-timestamp=true ! "
             << "video/x-h264,width=" << width << ",height=" << height
             << ",framerate=" << fps << "/1 ! "
             << "h264parse config-interval=-1 ! "
             << "nvv4l2decoder ! "
             << inferTail.str();
        candidates.push_back({"h264-nvdec-nv12", text.str()});
    }

    {
        std::ostringstream text;
        text << "v4l2src device=" << devicePath << " io-mode=2 do-timestamp=true ! "
             << "image/jpeg,width=" << width << ",height=" << height
             << ",framerate=" << fps << "/1 ! "
             << "jpegparse ! "
             << "jpegdec ! "
             << "videoconvert ! "
             << "video/x-raw,format=NV12,width=" << width
             << ",height=" << height << " ! "
             << "nvvideoconvert ! "
             << "video/x-raw(memory:NVMM),format=NV12 ! "
             << inferTail.str();
        candidates.push_back({"mjpeg-cpujpeg-nv12", text.str()});
    }

    if (app_config::kDeepStreamPreferH264Capture && candidates.size() >= 2) {
        std::rotate(candidates.begin(), candidates.begin() + 1, candidates.begin() + 2);
    }

    return candidates;
}

bool DeepStreamYolo::createAndStartPipeline(const PipelineDefinition& definition)
{
    currentPipelineName = definition.name;
    std::cout << "[DS][GST] pipeline mode=" << definition.name
              << ":\n" << definition.text << std::endl;

    GError* error = nullptr;
    pipeline = gst_parse_launch(definition.text.c_str(), &error);
    if (!pipeline) {
        std::cerr << "[DS][GST] gst_parse_launch failed: "
                  << (error ? error->message : "unknown") << std::endl;
        if (error) {
            g_error_free(error);
        }
        return false;
    }
    if (error) {
        std::cerr << "[DS][GST] warning: " << error->message << std::endl;
        g_error_free(error);
    }

    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
    GstElement* primary = gst_bin_get_by_name(GST_BIN(pipeline), "primary");
    if (!appsink || !primary) {
        std::cerr << "[DS][GST] appsink or primary nvinfer not found" << std::endl;
        close();
        return false;
    }

    GstPad* srcPad = gst_element_get_static_pad(primary, "src");
    if (srcPad) {
        gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER,
                          &DeepStreamYolo::tensorProbe, this, nullptr);
        gst_object_unref(srcPad);
    }
    gst_object_unref(primary);

    const GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        (void)checkBusErrors();
        std::cerr << "[DS][GST] failed to start pipeline" << std::endl;
        close();
        return false;
    }

    return true;
}

bool DeepStreamYolo::primePipeline()
{
    if (!isOpened()) {
        return false;
    }

    const int startupTimeoutMs = waitingForEngineBuild
        ? app_config::kDeepStreamEngineBuildStartupTimeoutMs
        : app_config::kDeepStreamNormalStartupProbeTimeoutMs;
    const int maxAttempts = std::max(1, startupTimeoutMs / 300);
    if (waitingForEngineBuild) {
        std::cout << "[DS][CONFIG] waiting for first TensorRT engine build (up to "
                  << (startupTimeoutMs / 1000) << " s)" << std::endl;
    }

    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        if (!checkBusErrors()) {
            return false;
        }

        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 300 * GST_MSECOND);
        if (!sample) {
            continue;
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        GstVideoInfo videoInfo;
        if (caps && gst_video_info_from_caps(&videoInfo, caps)) {
            actualWidth = GST_VIDEO_INFO_WIDTH(&videoInfo);
            actualHeight = GST_VIDEO_INFO_HEIGHT(&videoInfo);
        }
        gst_sample_unref(sample);
        waitingForEngineBuild = false;
        return true;
    }

    (void)checkBusErrors();
    std::cerr << "[DS][GST] no frame from pipeline during startup probe" << std::endl;
    return false;
}

bool DeepStreamYolo::open(const std::string& devicePath, int width, int height, int fps)
{
    close();
    fatalError = false;
    pipelineErrorText.clear();
    waitingForEngineBuild = false;
    if (!writeInferConfig()) {
        return false;
    }

    currentDevice = devicePath;
    requestedWidth = width;
    requestedHeight = height;
    requestedFps = fps;
    sourceWidth = width;
    sourceHeight = height;

    const auto candidates = buildPipelineCandidates(devicePath, width, height, fps);
    for (const PipelineDefinition& definition : candidates) {
        close();
        currentDevice = devicePath;
        requestedWidth = width;
        requestedHeight = height;
        requestedFps = fps;
        sourceWidth = width;
        sourceHeight = height;

        if (!createAndStartPipeline(definition)) {
            close();
            continue;
        }

        if (primePipeline()) {
            std::cout << "[DS][GST] selected pipeline mode="
                      << currentPipelineName << std::endl;
            return true;
        }

        std::cerr << "[DS][GST] pipeline mode failed before first frame: "
                  << definition.name << std::endl;
        close();
        if (fatalError) {
            return false;
        }
    }

    return false;
}

bool DeepStreamYolo::isOpened() const
{
    return pipeline != nullptr && appsink != nullptr;
}

bool DeepStreamYolo::hasFatalError() const
{
    return fatalError;
}

std::string DeepStreamYolo::lastErrorText() const
{
    return pipelineErrorText;
}

bool DeepStreamYolo::checkBusErrors()
{
    if (!pipeline) {
        return false;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    if (!bus) {
        return true;
    }

    bool ok = true;
    while (GstMessage* msg = gst_bus_pop_filtered(
               bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)))
    {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* error = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &error, &debug);
            std::cerr << "[DS][GST] error: "
                      << (error ? error->message : "unknown") << " debug="
                      << (debug ? debug : "") << std::endl;
            const std::string errorText = error ? error->message : "unknown";
            const std::string debugText = debug ? debug : "";
            pipelineErrorText = errorText + " " + debugText;
            const bool nvinferFailure =
                pipelineErrorText.find("GstNvInfer:primary") != std::string::npos ||
                pipelineErrorText.find("NVDSINFER") != std::string::npos ||
                pipelineErrorText.find("cudaErrorIllegalAddress") != std::string::npos;
            if (app_config::kDeepStreamStopOnNvinferError && nvinferFailure)
            {
                fatalError = true;
                std::cerr << "[DS][FATAL] nvinfer failed; switching camera modes "
                          << "cannot recover this pipeline. Check the compiled "
                          << "engine/input configuration and restart." << std::endl;
            }
            if (error) {
                g_error_free(error);
            }
            if (debug) {
                g_free(debug);
            }
            ok = false;
        } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
            std::cerr << "[DS][GST] EOS" << std::endl;
            ok = false;
        }
        gst_message_unref(msg);
    }

    gst_object_unref(bus);
    return ok;
}

bool DeepStreamYolo::read(cv::Mat& frame, bool trackingEnabled)
{
    if (!isOpened() || !checkBusErrors()) {
        return false;
    }

    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 500 * GST_MSECOND);
    if (!sample) {
        return false;
    }

    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo videoInfo;
    if (!caps || !gst_video_info_from_caps(&videoInfo, caps)) {
        gst_sample_unref(sample);
        return false;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!buffer || !gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    actualWidth = GST_VIDEO_INFO_WIDTH(&videoInfo);
    actualHeight = GST_VIDEO_INFO_HEIGHT(&videoInfo);
    const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&videoInfo, 0);
    cv::Mat wrapped(actualHeight, actualWidth, CV_8UC3, map.data, stride);
    frame = wrapped.clone();

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    std::vector<Detection> detections;
    cv::Size sourceFrameSize(sourceWidth > 0 ? sourceWidth : requestedWidth,
                             sourceHeight > 0 ? sourceHeight : requestedHeight);
    {
        std::lock_guard<std::mutex> lock(detectionsMutex);
        detections = latestDetections;
        if (latestSourceFrameSize.width > 0 && latestSourceFrameSize.height > 0) {
            sourceFrameSize = latestSourceFrameSize;
        }
    }

    processDetections(frame, detections, sourceFrameSize, trackingEnabled);
    return !frame.empty();
}

void DeepStreamYolo::close()
{
    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
    }
    if (appsink) {
        gst_object_unref(appsink);
        appsink = nullptr;
    }
    if (pipeline) {
        gst_object_unref(pipeline);
        pipeline = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(detectionsMutex);
        latestDetections.clear();
        latestSourceFrameSize = cv::Size();
    }
    actualWidth = 0;
    actualHeight = 0;
    sourceWidth = 0;
    sourceHeight = 0;
    currentPipelineName.clear();
}

std::string DeepStreamYolo::statusText() const
{
    std::ostringstream out;
    out << currentDevice << " DeepStream";
    if (!currentPipelineName.empty()) {
        out << "(" << currentPipelineName << ")";
    }
    out << " requested "
        << requestedWidth << "x" << requestedHeight << "@" << requestedFps;
    if (actualWidth > 0 && actualHeight > 0) {
        out << " display " << actualWidth << "x" << actualHeight << "@" << requestedFps;
    }
    if (sourceWidth > 0 && sourceHeight > 0) {
        out << " source " << sourceWidth << "x" << sourceHeight;
    }
    return out.str();
}

GstPadProbeReturn DeepStreamYolo::tensorProbe(GstPad*, GstPadProbeInfo* info, gpointer userData)
{
    auto* self = static_cast<DeepStreamYolo*>(userData);
    if (!self || !(info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
        return GST_PAD_PROBE_OK;
    }

    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (buffer) {
        self->updateDetectionsFromBuffer(buffer);
    }
    return GST_PAD_PROBE_OK;
}

void DeepStreamYolo::updateDetectionsFromBuffer(GstBuffer* buffer)
{
    NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batchMeta) {
        return;
    }

    std::vector<Detection> parsed;
    cv::Size parsedSourceSize(sourceWidth > 0 ? sourceWidth : requestedWidth,
                              sourceHeight > 0 ? sourceHeight : requestedHeight);
    for (NvDsMetaList* frameList = batchMeta->frame_meta_list;
         frameList != nullptr;
         frameList = frameList->next)
    {
        auto* frameMeta = static_cast<NvDsFrameMeta*>(frameList->data);
        if (!frameMeta) {
            continue;
        }

        // nvstreammux may expose its scaled surface size here. Tracking,
        // deadzones and FOV math must stay in physical camera coordinates.
        const int frameWidth = sourceWidth > 0
            ? sourceWidth
            : (frameMeta->source_frame_width > 0
                ? static_cast<int>(frameMeta->source_frame_width)
                : requestedWidth);
        const int frameHeight = sourceHeight > 0
            ? sourceHeight
            : (frameMeta->source_frame_height > 0
                ? static_cast<int>(frameMeta->source_frame_height)
                : requestedHeight);
        parsedSourceSize = cv::Size(frameWidth, frameHeight);

        for (NvDsMetaList* userList = frameMeta->frame_user_meta_list;
             userList != nullptr;
             userList = userList->next)
        {
            auto* userMeta = static_cast<NvDsUserMeta*>(userList->data);
            if (!userMeta ||
                userMeta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            {
                continue;
            }

            parsed = parseTensorMeta(userMeta->user_meta_data, frameWidth, frameHeight);
            break;
        }
    }

    std::lock_guard<std::mutex> lock(detectionsMutex);
    latestDetections = std::move(parsed);
    latestSourceFrameSize = parsedSourceSize;
}

std::vector<DeepStreamYolo::Detection> DeepStreamYolo::parseTensorMeta(void* tensorMetaPtr,
                                                                       int frameWidth,
                                                                       int frameHeight) const
{
    std::vector<Detection> detections;
    auto* tensorMeta = static_cast<NvDsInferTensorMeta*>(tensorMetaPtr);
    if (!tensorMeta || tensorMeta->num_output_layers == 0) {
        return detections;
    }

    NvDsInferLayerInfo& layer = tensorMeta->output_layers_info[0];
    const void* data = nullptr;
    if (tensorMeta->out_buf_ptrs_host && tensorMeta->out_buf_ptrs_host[0]) {
        data = tensorMeta->out_buf_ptrs_host[0];
    } else {
        data = layer.buffer;
    }
    if (!data) {
        return detections;
    }

    int rows = 0;
    int cols = 0;
    if (!inferRowsCols(layer.inferDims, rows, cols)) {
        std::cerr << "[DS][YOLO] unsupported output dims: numDims="
                  << layer.inferDims.numDims
                  << " numElements=" << layer.inferDims.numElements
                  << std::endl;
        return detections;
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int r = 0; r < rows; ++r) {
        const int base = r * cols;
        const float objConf = tensorValue(layer, data, base + 4);
        if (objConf < app_config::kDeepStreamConfThreshold) {
            continue;
        }

        int bestClass = -1;
        float bestScore = 0.0f;
        for (int c = 5; c < cols; ++c) {
            const float score = tensorValue(layer, data, base + c);
            if (score > bestScore) {
                bestScore = score;
                bestClass = c - 5;
            }
        }

        const float conf = bestScore * objConf;
        if (conf < app_config::kDeepStreamConfThreshold) {
            continue;
        }

        const float cx = tensorValue(layer, data, base + 0);
        const float cy = tensorValue(layer, data, base + 1);
        const float w = tensorValue(layer, data, base + 2);
        const float h = tensorValue(layer, data, base + 3);

        const float sx = static_cast<float>(frameWidth) /
                         static_cast<float>(app_config::kDeepStreamNetworkInputWidth);
        const float sy = static_cast<float>(frameHeight) /
                         static_cast<float>(app_config::kDeepStreamNetworkInputHeight);

        cv::Rect box(
            static_cast<int>(std::round((cx - 0.5f * w) * sx)),
            static_cast<int>(std::round((cy - 0.5f * h) * sy)),
            static_cast<int>(std::round(w * sx)),
            static_cast<int>(std::round(h * sy))
        );
        box &= cv::Rect(0, 0, frameWidth, frameHeight);
        if (box.area() <= 0) {
            continue;
        }

        boxes.push_back(box);
        confidences.push_back(conf);
        classIds.push_back(bestClass);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes,
                      confidences,
                      app_config::kDeepStreamConfThreshold,
                      app_config::kDeepStreamNmsThreshold,
                      keep);

    detections.reserve(keep.size());
    for (int idx : keep) {
        Detection detection;
        detection.classId = classIds[idx];
        detection.confidence = confidences[idx];
        detection.box = boxes[idx];
        detection.center = cv::Point(detection.box.x + detection.box.width / 2,
                                     detection.box.y + detection.box.height / 2);
        detection.area = detection.box.area();
        detections.push_back(detection);
    }
    return detections;
}

void DeepStreamYolo::processDetections(cv::Mat& frame,
                                       const std::vector<Detection>& detections,
                                       const cv::Size& sourceFrameSize,
                                       bool trackingEnabled)
{
    std::vector<Candidate> candidates;
    candidates.reserve(detections.size());

    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        Candidate candidate;
        candidate.index = i;
        candidate.detection = detections[i];
        candidate.score = targetScore(candidate.detection, haveLastTarget, lastTargetCenter);
        candidates.push_back(candidate);
    }

    int selected = -1;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i].score > bestScore) {
            bestScore = candidates[i].score;
            selected = i;
        }
    }

    const cv::Size displaySize(frame.cols, frame.rows);
    cv::Point selectedDisplayCenter;
    const cv::Point* selectedCenter = nullptr;
    if (selected >= 0) {
        selectedDisplayCenter = scalePointToDisplay(candidates[selected].detection.center,
                                                    sourceFrameSize,
                                                    displaySize);
        selectedCenter = &selectedDisplayCenter;
    }

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        const Detection& detection = candidates[i].detection;
        const bool isSelected = (i == selected);
        const cv::Scalar color = isSelected ? cv::Scalar(0, 255, 0)
                                            : cv::Scalar(0, 0, 255);
        const cv::Rect displayBox = scaleRectToDisplay(detection.box,
                                                       sourceFrameSize,
                                                       displaySize);

        cv::rectangle(frame, displayBox, color, isSelected ? 3 : 2);
        std::string label = cv::format("%.2f", detection.confidence);
        if (detection.classId >= 0 &&
            detection.classId < static_cast<int>(classes.size()))
        {
            label = classes[detection.classId] + ":" + label;
        }
        cv::putText(frame,
                    label,
                    cv::Point(displayBox.x, std::max(0, displayBox.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2);
    }

    if (trackingEnabled && selected >= 0) {
        const Detection& target = candidates[selected].detection;
        TargetManager::submitCameraTarget(target.center, target.box, sourceFrameSize);
        lastTargetCenter = target.center;
        haveLastTarget = true;
        lostTargetFrames = 0;
    } else {
        if (haveLastTarget && ++lostTargetFrames > TARGET_MEMORY_FRAMES) {
            haveLastTarget = false;
        }
        if (trackingEnabled) {
            TargetManager::submitCameraMiss();
        }
    }

    drawTrackingOverlay(frame, selectedCenter, sourceFrameSize);
}

void DeepStreamYolo::loadClasses()
{
    classes.clear();
    std::ifstream file(app_config::kYoloClassesPath);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classes.push_back(line);
        }
    }
}
