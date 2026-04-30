#include "my_yolo.h"
#include "target_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

my_yolo::my_yolo()
{
#if OPENCV_CPU_THREADS
    cv::setNumThreads(OPENCV_CPU_THREADS);
    cv::setUseOptimized(true);
#endif

    weightPath = "/home/nick/qt/yolo_quadro_weights/quadron_1280.onnx";

    if (weightPath.find("_1280") != std::string::npos)      yolo_width = yolo_height = 1280;
    else if (weightPath.find("_1920") != std::string::npos) yolo_width = yolo_height = 1920;
    else if (weightPath.find("_2560") != std::string::npos) yolo_width = yolo_height = 2560;
    else if (weightPath.find("_4000") != std::string::npos) yolo_width = yolo_height = 4000;
    else                                                    yolo_width = yolo_height = 640;

    CV_Assert(!weightPath.empty());
    CV_Assert(yolo_width > 0 && yolo_height > 0);

    net = cv::dnn::readNet(weightPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    net.enableFusion(false);
    net.enableWinograd(false);

    confThreshold = 0.3f;
    nmsThreshold  = 0.5f;

    scale  = 1.0/122.0;
    mean   = cv::Scalar(120,120,120);
    swapRB = true;

    loadClasses("/home/nick/qt/yolo_quadro_weights/quadro_3000.names");

    classIds.reserve(256);
    confidences.reserve(256);
    boxes.reserve(256);
    keep_idx.reserve(256);

    outs.reserve(3);

    gpu_resized_u8.create(yolo_height, yolo_width, CV_8UC3);
    gpu_blob_f32.create(yolo_height * 3, yolo_width, CV_32F);
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

    cv::Mat cpu_blob; // fallback

    // ============================
    // 2.1 blob (preprocess)
    // ============================
    tm_blob.start();
#if ENABLE_CUDA_PREPROCESS
    try {
        gpu_frame_u8.upload(frame, gpu_stream);

        cv::cuda::resize(gpu_frame_u8, gpu_resized_u8,
                         cv::Size(yolo_width, yolo_height),
                         0, 0, cv::INTER_LINEAR, gpu_stream);

        cudaStream_t s = cv::cuda::StreamAccessor::getStream(gpu_stream);

        bool ok = cuda_preprocess_bgr_to_nchw(
            gpu_resized_u8,
            gpu_blob_f32,
            yolo_width, yolo_height,
            (float)scale,
            (float)mean[0], (float)mean[1], (float)mean[2],
            swapRB,
            s
        );

        gpu_stream.waitForCompletion();
        if (!ok) throw std::runtime_error("cuda_preprocess_bgr_to_nchw failed");

        try {
            net.setInput(gpu_blob_f32);
        } catch (...) {
            gpu_blob_f32.download(cpu_blob, gpu_stream);
            gpu_stream.waitForCompletion();

            int sz[] = {1, 3, yolo_height, yolo_width};
            cv::Mat blob4d(4, sz, CV_32F, cpu_blob.ptr<float>());
            net.setInput(blob4d);
        }
    } catch (...) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, scale,
                                              cv::Size(yolo_width, yolo_height),
                                              mean, swapRB, false, CV_32F);
        net.setInput(blob);
    }
#else
    cv::Mat blob = cv::dnn::blobFromImage(frame, scale,
                                          cv::Size(yolo_width, yolo_height),
                                          mean, swapRB, false, CV_32F);
    net.setInput(blob);
#endif
    tm_blob.stop();

    // ============================
    // 2.2 forward (inference)
    // ============================
    tm_fwd.start();
    outs.clear();
    net.forward(outs, net.getUnconnectedOutLayersNames());
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
        TargetManager::submitCameraTarget(target.center, frame.size());

        last_target_center = target.center;
        have_last_target = true;
        lost_target_frames = 0;
    } else {
        if (have_last_target && ++lost_target_frames > TARGET_MEMORY_FRAMES) {
            have_last_target = false;
        }
        TargetManager::submitCameraMiss();
    }
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
