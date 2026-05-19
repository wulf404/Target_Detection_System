#include "my_yolo.h"
#include "auto_tracker.h"
#include "app_config.h"
#include "target_manager.h"

#include <NvInferPlugin.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

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
};

constexpr int TARGET_MEMORY_FRAMES = 15;
constexpr double CONF_SCORE = 1000.0;
constexpr double AREA_SCORE = 0.7;
constexpr double DIST_PENALTY = 0.6;
constexpr double STICKY_RADIUS_PX = 260.0;
constexpr double STICKY_BONUS = 180.0;

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
                   bool have_last_target,
                   const cv::Point& last_target_center)
{
    double score = static_cast<double>(candidate.confidence) * CONF_SCORE;
    score += std::sqrt(static_cast<double>(candidate.area)) * AREA_SCORE;

    if (have_last_target) {
        const double dx = static_cast<double>(candidate.center.x - last_target_center.x);
        const double dy = static_cast<double>(candidate.center.y - last_target_center.y);
        const double dist = std::hypot(dx, dy);
        score -= std::min(dist, 1000.0) * DIST_PENALTY;
        if (dist <= STICKY_RADIUS_PX) {
            score += STICKY_BONUS;
        }
    }

    return score;
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

yolo_output my_yolo::run(cv::Mat &frame)
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

    // ============================
    // 2.1 blob (preprocess)
    // ============================
    tm_blob.start();
    bool inputReady = false;
#if ENABLE_CUDA_PREPROCESS
    try {
        gpu_frame_u8.upload(frame, gpu_stream);

        cv::cuda::resize(gpu_frame_u8, gpu_resized_u8,
                         cv::Size(yolo_width, yolo_height),
                         0, 0, cv::INTER_LINEAR, gpu_stream);

        cudaStream_t s = cv::cuda::StreamAccessor::getStream(gpu_stream);

        bool ok = cuda_preprocess_bgr_to_nchw_ptr(
            gpu_resized_u8,
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
        cv::Mat blob = cv::dnn::blobFromImage(frame, scale,
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
    cv::Mat blob = cv::dnn::blobFromImage(frame, scale,
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
    const float sx = (float)frame.cols / (float)yolo_width;
    const float sy = (float)frame.rows / (float)yolo_height;
    tm_scale.stop();

    // ============================
    // 2.6 draw+pack
    // ============================
    tm_drawpack.start();
    int kept = 0;
    std::vector<TargetCandidate> candidates;
    candidates.reserve(keep_idx.size());

    for (int idx : keep_idx)
    {
        cv::Rect b = boxes[idx];
        b.x = (int)std::round(b.x * sx);
        b.y = (int)std::round(b.y * sy);
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
        candidate.score = targetScore(candidate, have_last_target, last_target_center);

        candidates.push_back(candidate);
        kept++;
    }

    int selected = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i].score > best_score) {
            best_score = candidates[i].score;
            selected = i;
        }
    }

    const cv::Point* selectedCenter = nullptr;
    if (selected >= 0) {
        selectedCenter = &candidates[selected].center;
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

    if (selected >= 0) {
        const TargetCandidate& target = candidates[selected];
        TargetManager::submitCameraTarget(target.center, target.box, frame.size());

        last_target_center = target.center;
        have_last_target = true;
        lost_target_frames = 0;
    } else {
        if (have_last_target && ++lost_target_frames > TARGET_MEMORY_FRAMES) {
            have_last_target = false;
        }
        TargetManager::submitCameraMiss();
    }

    drawTrackingOverlay(frame, selectedCenter);
    tm_drawpack.stop();

    tm_total.stop();

    // ============================
    // PRINT PERF (каждые N вызовов run())
    // но делим на perf_divisor, чтобы это было “на кадр” при пропуске инференса
    // ============================
    frame_counter++;
    if (frame_counter % perf_print_every == 0)
    {
        const double div = (double)perf_divisor;

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
