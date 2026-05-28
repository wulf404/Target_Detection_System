#include "camera.h"
#include "app_config.h"

#include <QFile>
#include <QString>
#include <QTimer>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <sstream>

namespace {
namespace fs = std::filesystem;

struct VideoDeviceInfo
{
    std::string devPath;
    std::string name;
    std::string vid;
    std::string pid;
    int index = 0;
    int score = 0;
};

std::string trim(std::string value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string readFirstLine(const fs::path& path)
{
    std::ifstream file(path);
    std::string line;
    if (file && std::getline(file, line)) {
        return trim(line);
    }
    return {};
}

std::string normalizeHexId(std::string value)
{
    value = lowerCopy(trim(value));
    if (value.rfind("0x", 0) == 0) {
        value.erase(0, 2);
    }
    return value;
}

bool containsInsensitive(const std::string& text, const std::string& needle)
{
    if (needle.empty()) {
        return false;
    }
    return lowerCopy(text).find(lowerCopy(needle)) != std::string::npos;
}

std::string readFromAncestor(const fs::path& start, const char* fileName)
{
    std::error_code ec;
    fs::path current = fs::weakly_canonical(start, ec);
    if (ec) {
        current = start;
    }

    while (!current.empty()) {
        const std::string value = readFirstLine(current / fileName);
        if (!value.empty()) {
            return value;
        }
        if (current == current.root_path()) {
            break;
        }
        current = current.parent_path();
    }
    return {};
}

int parseVideoIndex(const std::string& name)
{
    if (name.rfind("video", 0) != 0) {
        return 9999;
    }
    try {
        return std::stoi(name.substr(5));
    } catch (...) {
        return 9999;
    }
}

int cameraPreferenceScore(const VideoDeviceInfo& info)
{
    int score = 0;
    const std::string preferredVid = normalizeHexId(app_config::kCameraPreferredVid);
    const std::string preferredPid = normalizeHexId(app_config::kCameraPreferredPid);
    const std::string preferredName = app_config::kCameraPreferredNameContains;

    if (!preferredVid.empty() && info.vid == preferredVid) {
        score += 40;
    }
    if (!preferredPid.empty() && info.pid == preferredPid) {
        score += 40;
    }
    if (!preferredVid.empty() && !preferredPid.empty() &&
        info.vid == preferredVid && info.pid == preferredPid)
    {
        score += 80;
    }
    if (containsInsensitive(info.name, preferredName)) {
        score += 50;
    }
    return score;
}

VideoDeviceInfo readVideoDeviceInfo(const std::string& devPath)
{
    VideoDeviceInfo info;
    info.devPath = devPath;

    const std::string nodeName = fs::path(devPath).filename().string();
    info.index = parseVideoIndex(nodeName);

    const fs::path sysPath = fs::path("/sys/class/video4linux") / nodeName;
    info.name = readFirstLine(sysPath / "name");

    const fs::path devicePath = sysPath / "device";
    info.vid = normalizeHexId(readFromAncestor(devicePath, "idVendor"));
    info.pid = normalizeHexId(readFromAncestor(devicePath, "idProduct"));
    info.score = cameraPreferenceScore(info);
    return info;
}

std::string describeVideoDevice(const VideoDeviceInfo& info,
                                int reqWidth,
                                int reqHeight,
                                int reqFps,
                                int actualWidth = 0,
                                int actualHeight = 0,
                                double actualFps = 0.0)
{
    std::ostringstream out;
    out << info.devPath;
    if (!info.name.empty()) {
        out << " " << info.name;
    }
    if (!info.vid.empty() || !info.pid.empty()) {
        out << " [" << (info.vid.empty() ? "????" : info.vid)
            << ":" << (info.pid.empty() ? "????" : info.pid) << "]";
    }
    out << " requested " << reqWidth << "x" << reqHeight << "@" << reqFps;
    if (actualWidth > 0 && actualHeight > 0) {
        const int fpsRounded = actualFps > 0.5
            ? static_cast<int>(std::round(actualFps))
            : reqFps;
        out << " actual " << actualWidth << "x" << actualHeight << "@" << fpsRounded;
    }
    return out.str();
}

std::vector<VideoDeviceInfo> scanVideoDevices()
{
    std::vector<VideoDeviceInfo> devices;
    std::error_code ec;
    const fs::path root("/sys/class/video4linux");
    if (!fs::exists(root, ec)) {
        return devices;
    }

    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (ec) {
            break;
        }
        const std::string nodeName = entry.path().filename().string();
        if (nodeName.rfind("video", 0) != 0) {
            continue;
        }

        const std::string devPath = "/dev/" + nodeName;
        if (!QFile::exists(QString::fromStdString(devPath))) {
            continue;
        }
        devices.push_back(readVideoDeviceInfo(devPath));
    }

    std::sort(devices.begin(), devices.end(),
              [](const VideoDeviceInfo& a, const VideoDeviceInfo& b) {
                  if (a.score != b.score) {
                      return a.score > b.score;
                  }
                  return a.index < b.index;
              });
    return devices;
}
} // namespace

// ============================================================
// USB GStreamer pipeline
// Для ELP 4K USB camera.
// Приоритет: MJPEG 4K60.
// ============================================================
std::string Camera::buildUsbPipeline(const std::string& devicePath) const
{
    return cv::format(
        "v4l2src device=%s io-mode=2 ! "
        "image/jpeg,width=%d,height=%d,framerate=%d/1 ! "
        "jpegparse ! "
        "jpegdec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false",
        devicePath.c_str(),
        camera_width, camera_height, camera_fps
    );
}

std::vector<std::string> Camera::cameraDeviceCandidates() const
{
    const std::string overridePath = app_config::kCameraDevicePathOverride;
    std::vector<std::string> devices;

    if (!overridePath.empty()) {
        if (QFile::exists(QString::fromStdString(overridePath))) {
            devices.push_back(overridePath);
        }
        return devices;
    }

    const auto scannedDevices = scanVideoDevices();
    for (const VideoDeviceInfo& info : scannedDevices) {
        std::cout << "[CAM][USB][SCAN] "
                  << describeVideoDevice(info, camera_width, camera_height, camera_fps)
                  << " score=" << info.score
                  << std::endl;
        devices.push_back(info.devPath);
    }

    if (devices.empty()) {
        for (int i = 0; i < 10; ++i) {
            const std::string path = "/dev/video" + std::to_string(i);
            if (QFile::exists(QString::fromStdString(path))) {
                devices.push_back(path);
            }
        }
    }

    return devices;
}

void Camera::publishDeviceState(bool connected, const std::string& description)
{
    const QString text = description.empty()
        ? QString("not found")
        : QString::fromStdString(description);

    if (camera_device_connected == connected &&
        camera_device_description == text)
    {
        return;
    }

    camera_device_connected = connected;
    camera_device_description = text;
    emit deviceStateChanged(connected, text);
}

Camera::Camera([[maybe_unused]] QObject *parent, int width)
{
    CV_Assert(width > 100);

    setStopped(true);
    setStarted(false);

    camera_width = width;
    applyUsbDefaults();

    Yolo = new my_yolo;
    Yolo->setPerfPrintEvery(30);
}

Camera::~Camera()
{
    closeDevice();
    delete Yolo;
}

void Camera::setRequestedWidth(int width)
{
    camera_width = width;
    applyUsbDefaults();
}

void Camera::applyUsbDefaults()
{
    // Под режимы, которые ты уже показал у камеры:
    // MJPG: 3840x2160 @ 60
    //       2560x1440 @ 60
    //       1920x1080 @ 60
    //       1280x720  @ 60
    //       640x480   @ 60

    if (camera_width >= 3840) {
        camera_width  = 3840;
        camera_height = 2160;
        camera_fps    = 60;
    } else if (camera_width >= 2560) {
        camera_width  = 2560;
        camera_height = 1440;
        camera_fps    = 60;
    } else if (camera_width >= 1920) {
        camera_width  = 1920;
        camera_height = 1080;
        camera_fps    = 60;
    } else if (camera_width >= 1280) {
        camera_width  = 1280;
        camera_height = 720;
        camera_fps    = 60;
    } else {
        camera_width  = 640;
        camera_height = 480;
        camera_fps    = 60;
    }
}

bool Camera::openDevice()
{
    try
    {
        const auto devices = cameraDeviceCandidates();
        if (devices.empty()) {
            std::cerr << "[CAM][USB][GST] no /dev/video* camera candidates" << std::endl;
            publishDeviceState(false, "not found");
            return false;
        }
        publishDeviceState(true, describeVideoDevice(readVideoDeviceInfo(devices.front()),
                                                     camera_width,
                                                     camera_height,
                                                     camera_fps));

        if (video.isOpened()) {
            video.release();
        }

        for (const std::string& devicePath : devices) {
            const std::string pipeline = buildUsbPipeline(devicePath);

            std::cout << "[CAM][USB][GST] try open " << devicePath
                      << " pipeline:\n" << pipeline << std::endl;

            if (!video.open(pipeline, cv::CAP_GSTREAMER)) {
                std::cerr << "[CAM][USB][GST] open failed: " << devicePath << std::endl;
                continue;
            }

#ifdef CV_CAP_PROP_BUFFERSIZE
            video.set(cv::CAP_PROP_BUFFERSIZE, 1);
#endif

            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            cv::Mat test;
            for (int i = 0; i < 20; ++i) {
                if (video.read(test) && !test.empty()) {
                    current_device_path = devicePath;
                    const double actualFps = video.get(cv::CAP_PROP_FPS);
                    publishDeviceState(true, describeVideoDevice(readVideoDeviceInfo(current_device_path),
                                                                 camera_width,
                                                                 camera_height,
                                                                 camera_fps,
                                                                 test.cols,
                                                                 test.rows,
                                                                 actualFps));
                    std::cout << "[CAM][USB][GST] opened OK: "
                              << current_device_path << " "
                              << test.cols << "x" << test.rows
                              << ", ch=" << test.channels()
                              << ", req=" << camera_width << "x" << camera_height
                              << "@" << camera_fps
                              << ", actual_fps=" << actualFps
                              << std::endl;
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            std::cerr << "[CAM][USB][GST] open ok, but no frames: "
                      << devicePath << std::endl;
            video.release();
        }

        current_device_path.clear();
        publishDeviceState(true, "open failed");
        return false;
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "[CAM][USB][GST][OpenCV EXCEPTION] " << e.what() << std::endl;
        if (video.isOpened()) video.release();
        current_device_path.clear();
        publishDeviceState(true, "OpenCV exception");
        return false;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CAM][USB][GST][std EXCEPTION] " << e.what() << std::endl;
        if (video.isOpened()) video.release();
        current_device_path.clear();
        publishDeviceState(true, "std exception");
        return false;
    }
}

void Camera::closeDevice()
{
    stopCaptureThread();

    if (video.isOpened()) {
        video.release();
        std::cout << "[CAM] device closed" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    current_device_path.clear();
}

void Camera::Stop()
{
    TaskRunner::Stop();
    capture_running.store(false);
    latest_frame_cv.notify_all();
}

void Camera::startCaptureThread()
{
    stopCaptureThread();

    {
        std::lock_guard<std::mutex> lock(latest_frame_mutex);
        latest_frame.release();
        latest_capture_ms = 0.0;
        latest_sequence = 0;
    }

    capture_failed.store(false);
    capture_running.store(true);

    capture_thread = std::thread([this]() {
        int bad_reads_in_row = 0;
        constexpr int kBadReadsLimit = 5;

        while (capture_running.load() && !isStopped()) {
            cv::TickMeter tm_read;
            cv::Mat frame;
            bool ok = false;

            try {
                tm_read.start();
                ok = video.read(frame);
                tm_read.stop();
            } catch (const cv::Exception& e) {
                tm_read.stop();
                std::cerr << "[CAM][CAPTURE][OpenCV] " << e.what() << std::endl;
                ok = false;
            } catch (const std::exception& e) {
                tm_read.stop();
                std::cerr << "[CAM][CAPTURE][std] " << e.what() << std::endl;
                ok = false;
            }

            if (!ok || frame.empty()) {
                ++bad_reads_in_row;
                if (bad_reads_in_row >= kBadReadsLimit) {
                    capture_failed.store(true);
                    capture_running.store(false);
                    latest_frame_cv.notify_all();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            bad_reads_in_row = 0;

            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
            } else if (frame.channels() == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }

            {
                std::lock_guard<std::mutex> lock(latest_frame_mutex);
                latest_frame = std::move(frame);
                latest_capture_ms = tm_read.getTimeMilli();
                ++latest_sequence;
            }
            latest_frame_cv.notify_one();
        }
    });
}

void Camera::stopCaptureThread()
{
    capture_running.store(false);
    latest_frame_cv.notify_all();

    if (capture_thread.joinable()) {
        capture_thread.join();
    }

    capture_failed.store(false);
}

bool Camera::takeLatestFrame(cv::Mat& frame,
                             double& captureMs,
                             std::uint64_t& sequence,
                             std::uint64_t lastSequence,
                             int timeoutMs)
{
    std::unique_lock<std::mutex> lock(latest_frame_mutex);
    const bool ready = latest_frame_cv.wait_for(
        lock,
        std::chrono::milliseconds(std::max(1, timeoutMs)),
        [this, lastSequence]() {
            return isStopped() ||
                   capture_failed.load() ||
                   latest_sequence != lastSequence;
        });

    if (!ready || isStopped() || latest_sequence == lastSequence || latest_frame.empty()) {
        return false;
    }

    frame = latest_frame;
    captureMs = latest_capture_ms;
    sequence = latest_sequence;
    return true;
}

void Camera::runSequentialPipeline()
{
    try
    {
        setStopped(false);
        setStarted(true);

        while (!isStopped() && !openDevice()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        if (isStopped()) {
            closeDevice();
            setStopped(true);
            setStarted(false);
            return;
        }

        if (!video.isOpened()) {
            std::cerr << "[CAM] start failed: device not opened" << std::endl;
            setStopped(true);
            setStarted(false);
            return;
        }

        if (Yolo) {
            Yolo->setPerfDivisor(infer_every_n);
        }

        const cv::Size out_size(1280, 720);
        const int perf_every = 60;

        double sum_read = 0.0;
        double sum_yolo = 0.0;
        double sum_vis  = 0.0;
        double sum_emit = 0.0;

        int perf_frames = 0;
        int perf_yolo_runs = 0;

        double fps_ema = 0.0;
        const double fps_alpha = 0.15;
        int64 t_prev = cv::getTickCount();

        int bad_reads_in_row = 0;
        const int BAD_READS_LIMIT = 5;

        int frame_id = 0;

        while (!isStopped())
        {
            cv::TickMeter tm_read, tm_yolo, tm_vis, tm_emit;
            cv::Mat frame;

            // ========================================================
            // 1) read
            // ========================================================
            tm_read.start();
            const bool ok = video.read(frame);
            tm_read.stop();

            if (!ok || frame.empty())
            {
                bad_reads_in_row++;

                if (bad_reads_in_row >= BAD_READS_LIMIT)
                {
                    std::cerr << "[CAM][USB][GST] stream lost. Reopening..." << std::endl;
                    closeDevice();
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));

                    while (!isStopped()) {
                        if (openDevice()) {
                            std::cerr << "[CAM][USB][GST] reconnected OK\n";
                            bad_reads_in_row = 0;
                            t_prev = cv::getTickCount();
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(300));
                    }

                    if (isStopped()) {
                        break;
                    }
                    continue;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            bad_reads_in_row = 0;

            // ========================================================
            // 2) подстраховка по формату
            // ========================================================
            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
            } else if (frame.channels() == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }

            // ========================================================
            // 3) FPS
            // ========================================================
            int64 t_now = cv::getTickCount();
            const double dt = static_cast<double>(t_now - t_prev) / cv::getTickFrequency();
            t_prev = t_now;

            const double fps_inst = (dt > 1e-9) ? (1.0 / dt) : 0.0;
            if (fps_ema <= 0.0) fps_ema = fps_inst;
            else fps_ema = fps_alpha * fps_inst + (1.0 - fps_alpha) * fps_ema;

            // ========================================================
            // 4) YOLO
            // ========================================================
            const bool do_infer = yolo_enabled &&
                                  (infer_every_n <= 1 || ((frame_id % infer_every_n) == 0));

            if (do_infer && Yolo)
            {
                tm_yolo.start();
                (void)Yolo->run(frame, tm_read.getTimeMilli());
                tm_yolo.stop();
                perf_yolo_runs++;
            }

            // ========================================================
            // 5) vis
            // ========================================================
            tm_vis.start();

            cv::resize(frame, frame, out_size);

            std::string infer_text = yolo_enabled
                ? ("TRT 1/" + std::to_string(infer_every_n))
                : "OFF";

            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "FPS: %.1f  infer: %s  src: USB-GST  %dx%d@%d",
                          fps_ema,
                          infer_text.c_str(),
                          camera_width,
                          camera_height,
                          camera_fps);

            cv::putText(frame, buf, cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            tm_vis.stop();

            // ========================================================
            // 6) emit
            // ========================================================
            tm_emit.start();
            emit output_frame(frame);
            tm_emit.stop();

            // ========================================================
            // PERF
            // ========================================================
            sum_read += tm_read.getTimeMilli();
            sum_yolo += tm_yolo.getTimeMilli();
            sum_vis  += tm_vis.getTimeMilli();
            sum_emit += tm_emit.getTimeMilli();
            perf_frames++;

            if (perf_frames >= perf_every)
            {
                const double avg_read = sum_read / perf_frames;
                const double avg_yolo = sum_yolo / perf_frames;
                const double avg_vis  = sum_vis  / perf_frames;
                const double avg_emit = sum_emit / perf_frames;

                std::cerr
                    << "[PERF CAM] "
                    << "1.read=" << avg_read << "ms, "
                    << "2.yolo(avg)=" << avg_yolo << "ms, "
                    << "3.vis=" << avg_vis << "ms, "
                    << "4.emit=" << avg_emit << "ms, "
                    << "infer_runs=" << perf_yolo_runs << "/" << perf_frames
                    << "\n";

                sum_read = sum_yolo = sum_vis = sum_emit = 0.0;
                perf_frames = 0;
                perf_yolo_runs = 0;
            }

            frame_id++;
        }
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "[CAM][EXCEPTION][OpenCV] " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CAM][EXCEPTION][std] " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[CAM][EXCEPTION] unknown" << std::endl;
    }

    closeDevice();
    setStopped(true);
    setStarted(false);
}

void Camera::runLatestFramePipeline()
{
    try
    {
        setStopped(false);
        setStarted(true);

        while (!isStopped())
        {
            while (!isStopped() && !openDevice()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }

            if (isStopped()) {
                break;
            }

            if (!video.isOpened()) {
                std::cerr << "[CAM] start failed: device not opened" << std::endl;
                break;
            }

            if (Yolo) {
                Yolo->setPerfDivisor(infer_every_n);
            }

            startCaptureThread();
            std::cout << "[CAM][PIPE] latest-frame pipeline enabled" << std::endl;

            const cv::Size out_size(1280, 720);
            const int perf_every = 60;

            double sum_wait = 0.0;
            double sum_read = 0.0;
            double sum_yolo = 0.0;
            double sum_vis  = 0.0;
            double sum_emit = 0.0;

            int perf_frames = 0;
            int perf_yolo_runs = 0;
            std::uint64_t perf_dropped_frames = 0;

            double fps_ema = 0.0;
            const double fps_alpha = 0.15;
            int64 t_prev = cv::getTickCount();

            int frame_id = 0;
            std::uint64_t last_sequence = 0;

            while (!isStopped() && !capture_failed.load())
            {
                cv::TickMeter tm_wait, tm_yolo, tm_vis, tm_emit;
                cv::Mat frame;
                double captureMs = 0.0;
                std::uint64_t sequence = 0;

                tm_wait.start();
                const bool gotFrame = takeLatestFrame(
                    frame,
                    captureMs,
                    sequence,
                    last_sequence,
                    app_config::kCameraFrameWaitTimeoutMs);
                tm_wait.stop();

                if (!gotFrame) {
                    if (capture_failed.load()) {
                        break;
                    }
                    continue;
                }

                if (last_sequence > 0 && sequence > last_sequence + 1) {
                    perf_dropped_frames += sequence - last_sequence - 1;
                }
                last_sequence = sequence;

                if (frame.channels() == 4) {
                    cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
                } else if (frame.channels() == 1) {
                    cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
                }

                int64 t_now = cv::getTickCount();
                const double dt = static_cast<double>(t_now - t_prev) / cv::getTickFrequency();
                t_prev = t_now;

                const double fps_inst = (dt > 1e-9) ? (1.0 / dt) : 0.0;
                if (fps_ema <= 0.0) fps_ema = fps_inst;
                else fps_ema = fps_alpha * fps_inst + (1.0 - fps_alpha) * fps_ema;

                const bool do_infer = yolo_enabled &&
                                      (infer_every_n <= 1 || ((frame_id % infer_every_n) == 0));

                if (do_infer && Yolo)
                {
                    tm_yolo.start();
                    (void)Yolo->run(frame, captureMs);
                    tm_yolo.stop();
                    perf_yolo_runs++;
                }

                tm_vis.start();

                cv::resize(frame, frame, out_size);

                std::string infer_text = yolo_enabled
                    ? ("TRT 1/" + std::to_string(infer_every_n))
                    : "OFF";

                char buf[240];
                std::snprintf(buf, sizeof(buf),
                              "FPS: %.1f  infer: %s  src: USB-GST latest  %dx%d@%d  drop:%llu",
                              fps_ema,
                              infer_text.c_str(),
                              camera_width,
                              camera_height,
                              camera_fps,
                              static_cast<unsigned long long>(perf_dropped_frames));

                cv::putText(frame, buf, cv::Point(15, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

                tm_vis.stop();

                tm_emit.start();
                emit output_frame(frame);
                tm_emit.stop();

                sum_wait += tm_wait.getTimeMilli();
                sum_read += captureMs;
                sum_yolo += tm_yolo.getTimeMilli();
                sum_vis  += tm_vis.getTimeMilli();
                sum_emit += tm_emit.getTimeMilli();
                perf_frames++;

                if (perf_frames >= perf_every)
                {
                    const double avg_wait = sum_wait / perf_frames;
                    const double avg_read = sum_read / perf_frames;
                    const double avg_yolo = sum_yolo / perf_frames;
                    const double avg_vis  = sum_vis  / perf_frames;
                    const double avg_emit = sum_emit / perf_frames;

                    std::cerr
                        << "[PERF CAM] mode=latest "
                        << "0.wait=" << avg_wait << "ms, "
                        << "1.read=" << avg_read << "ms, "
                        << "2.yolo(avg)=" << avg_yolo << "ms, "
                        << "3.vis=" << avg_vis << "ms, "
                        << "4.emit=" << avg_emit << "ms, "
                        << "infer_runs=" << perf_yolo_runs << "/" << perf_frames
                        << ", dropped=" << perf_dropped_frames
                        << "\n";

                    sum_wait = sum_read = sum_yolo = sum_vis = sum_emit = 0.0;
                    perf_frames = 0;
                    perf_yolo_runs = 0;
                    perf_dropped_frames = 0;
                }

                frame_id++;
            }

            stopCaptureThread();

            if (!isStopped()) {
                std::cerr << "[CAM][USB][GST] stream lost. Reopening..." << std::endl;
                closeDevice();
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
        }
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "[CAM][EXCEPTION][OpenCV] " << e.what() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CAM][EXCEPTION][std] " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[CAM][EXCEPTION] unknown" << std::endl;
    }

    closeDevice();
    setStopped(true);
    setStarted(false);
}

void Camera::run()
{
    if (app_config::kCameraLatestFramePipelineEnabled) {
        runLatestFramePipeline();
    } else {
        runSequentialPipeline();
    }
}

void Camera::checkAndReconnect()
{
    if (isStopped()) {
        return;
    }

    if (capture_running.load()) {
        if (!reconnectTimer) reconnectTimer = new QTimer(this);
        reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
        return;
    }

    if (cameraDeviceCandidates().empty()) {
        std::cerr << "[CAM][USB][GST] no camera device, waiting..." << std::endl;
        if (!reconnectTimer) reconnectTimer = new QTimer(this);
        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
        return;
    }

    if (video.isOpened()) {
        cv::Mat test;
        try {
            if (!video.read(test) || test.empty()) {
                std::cerr << "[CAM][USB][GST] stream lost, reopening..." << std::endl;
                video.release();
            } else {
                if (!reconnectTimer) reconnectTimer = new QTimer(this);
                reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
                return;
            }
        } catch (const cv::Exception &e) {
            std::cerr << "[CAM][USB][GST] exception in reconnect check: "
                      << e.what() << std::endl;
            video.release();
        }
    }

    if (openDevice()) {
        std::cout << "[CAM][USB][GST] reconnected successfully!" << std::endl;
        if (!reconnectTimer) reconnectTimer = new QTimer(this);
        reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
    } else {
        std::cerr << "[CAM][USB][GST] reconnect failed, retry later..." << std::endl;
        if (!reconnectTimer) reconnectTimer = new QTimer(this);
        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
    }
}

//#include "camera.h"
//#include <QFile>
//#include <QDir>
//#include <QFileInfo>
//#include <opencv2/videoio.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//#include <chrono>
//#include <thread>

//static void tuneCapture(cv::VideoCapture &cap, int width, int height, int fps)
//{
//    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
//    cap.set(cv::CAP_PROP_FPS, fps);
//    cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
//    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

//#ifdef CV_CAP_PROP_BUFFERSIZE
//    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
//#endif
//}

//Camera::Camera([[maybe_unused]] QObject *parent, int width)
//{
//    CV_Assert(width > 100);

//    setStopped(true);
//    setStarted(false);

//    camera_width = width;

//    if (camera_width == 1920) {
//        camera_height = 1080;
//        camera_fps    = 60;
//    } else if (camera_width == 3840) {
//        camera_height = 2160;
//        camera_fps    = 30;
//    } else if (camera_width == 4032) {
//        camera_height = 3040;
//        camera_fps    = 20;
//    } else {
//        camera_height = 1080;
//        camera_fps    = 30;
//    }

//    Yolo = new my_yolo;
//    Yolo->setPerfPrintEvery(30);
//}

//Camera::~Camera()
//{
//    closeDevice();
//    delete Yolo;
//}

//bool Camera::openDevice()
//{
//    const std::string devPath = "/dev/video0";
//    std::cout << "[CAM] try open: " << devPath << std::endl;

//    if (!video.open(devPath, cv::CAP_V4L2)) {
//        std::cerr << "[CAM] open failed: " << devPath << std::endl;
//        return false;
//    }

//    tuneCapture(video, camera_width, camera_height, camera_fps);

//    cv::Mat test;
//    if (!video.read(test) || test.empty()) {
//        std::cerr << "[CAM] open succeeded, but no frames. Reopen…" << std::endl;
//        video.release();
//        return false;
//    }

//    std::cout << "[CAM] opened OK: " << devPath
//              << " (" << test.cols << "x" << test.rows << ")" << std::endl;
//    return true;
//}

//void Camera::closeDevice()
//{
//    if (video.isOpened()) {
//        video.release();
//        std::cout << "[CAM] device closed" << std::endl;
//    }
//}

//void Camera::run()
//{
//    setStopped(false);
//    setStarted(true);

//    for (int attempt = 0; attempt < 20 && !openDevice(); ++attempt) {
//        std::this_thread::sleep_for(std::chrono::milliseconds(300));
//    }
//    if (!video.isOpened()) {
//        std::cerr << "[CAM] start failed: device not opened" << std::endl;
//        setStopped(true);
//        setStarted(false);
//        return;
//    }

//    // PERF делитель для YOLO-логов (чтобы YOLO показывал “на кадр”, а не “на запуск”)
//    if (Yolo) Yolo->setPerfDivisor(infer_every_n);

//    cv::Size out_size(1280, 720);

//    // печатаем раз в N кадров окна
//    const int perf_every = 60;

//    // накапливаем суммы, чтобы печатать среднее “на кадр”
//    double sum_read = 0, sum_yolo = 0, sum_vis = 0, sum_emit = 0;
//    int perf_frames = 0;
//    int perf_yolo_runs = 0;

//    // FPS EMA
//    double fps_ema = 0.0;
//    const double fps_alpha = 0.15;
//    int64 t_prev = cv::getTickCount();

//    // reconnect
//    int bad_reads_in_row = 0;
//    const int BAD_READS_LIMIT = 5;

//    int frame_id = 0;

//    while (!isStopped())
//    {
//        cv::TickMeter tm_read, tm_yolo, tm_vis, tm_emit;

//        // 1.read
//        tm_read.start();
//        cv::Mat frame;
//        bool ok = video.read(frame);
//        tm_read.stop();

//        if (!ok || frame.empty())
//        {
//            bad_reads_in_row++;
//            if (bad_reads_in_row >= BAD_READS_LIMIT)
//            {
//                std::cerr << "[CAM] stream lost. Reopening...\n";
//                closeDevice();
//                std::this_thread::sleep_for(std::chrono::milliseconds(200));

//                bool reopened = false;
//                for (int t = 0; t < 50 && !isStopped(); ++t) {
//                    if (openDevice()) { reopened = true; break; }
//                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
//                }
//                if (reopened) {
//                    std::cerr << "[CAM] reconnected OK\n";
//                    bad_reads_in_row = 0;
//                    t_prev = cv::getTickCount();
//                } else {
//                    bad_reads_in_row = 0;
//                }
//                continue;
//            }

//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//            continue;
//        }

//        bad_reads_in_row = 0;

//        // FPS
//        int64 t_now = cv::getTickCount();
//        double dt = (double)(t_now - t_prev) / cv::getTickFrequency();
//        t_prev = t_now;

//        double fps_inst = (dt > 1e-9) ? (1.0 / dt) : 0.0;
//        if (fps_ema <= 0.0) fps_ema = fps_inst;
//        else fps_ema = fps_alpha * fps_inst + (1.0 - fps_alpha) * fps_ema;

//        // 2.yolo (раз в infer_every_n кадров)
//        const bool do_infer = (infer_every_n <= 1) || ((frame_id % infer_every_n) == 0);

//        if (do_infer)
//        {
//            tm_yolo.start();
//            (void)Yolo->run(frame);
//            tm_yolo.stop();
//            perf_yolo_runs++;
//        }
//        // иначе — просто показываем “сырой” кадр без обновления детекта

//        // 3.vis
//        tm_vis.start();
//        cv::resize(frame, frame, out_size);

//        char buf[160];
//        std::snprintf(buf, sizeof(buf),
//                      "FPS: %.1f  infer: 1/%d", fps_ema, infer_every_n);
//        cv::putText(frame, buf, cv::Point(15, 30),
//                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
//        tm_vis.stop();

//        // 4.emit_frame
//        tm_emit.start();
//        emit output_frame(frame);
//        tm_emit.stop();

//        // ===== PERF accumulation (среднее на кадр) =====
//        sum_read += tm_read.getTimeMilli();
//        sum_yolo += tm_yolo.getTimeMilli(); // тут 0 на пропущенных кадрах — это и есть “деление”
//        sum_vis  += tm_vis.getTimeMilli();
//        sum_emit += tm_emit.getTimeMilli();
//        perf_frames++;

//        if (perf_frames >= perf_every)
//        {
//            const double avg_read = sum_read / perf_frames;
//            const double avg_yolo = sum_yolo / perf_frames; // ключевой момент: честное “на кадр”
//            const double avg_vis  = sum_vis  / perf_frames;
//            const double avg_emit = sum_emit / perf_frames;

////            std::cerr
////                << "[PERF CAM0] "
////                << "1.read=" << avg_read << "ms, "
////                << "2.yolo(avg)=" << avg_yolo << "ms, "
////                << "3.vis=" << avg_vis << "ms, "
////                << "4.emit_frame=" << avg_emit << "ms, "
////                << "infer_runs=" << perf_yolo_runs << "/" << perf_frames
////                << "\n";

//            // reset
//            sum_read = sum_yolo = sum_vis = sum_emit = 0.0;
//            perf_frames = 0;
//            perf_yolo_runs = 0;
//        }

//        frame_id++;
//    }

//    closeDevice();
//    setStarted(false);
//}

//void Camera::checkAndReconnect()
//{
//    if (isStopped()) return;

//    const std::string devPath = "/dev/video0";
//    if (!QFile::exists(QString::fromStdString(devPath))) {
//        std::cerr << "[CAM] device missing, waiting..." << std::endl;
//        if (!reconnectTimer) reconnectTimer = new QTimer(this);
//        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
//        return;
//    }

//    if (video.isOpened()) {
//        cv::Mat test;
//        if (!video.read(test) || test.empty()) {
//            std::cerr << "[CAM] stream lost, reopening..." << std::endl;
//            video.release();
//        } else {
//            if (!reconnectTimer) reconnectTimer = new QTimer(this);
//            reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
//            return;
//        }
//    }

//    if (openDevice()) {
//        std::cout << "[CAM] reconnected successfully!" << std::endl;
//        if (!reconnectTimer) reconnectTimer = new QTimer(this);
//        reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
//    } else {
//        std::cerr << "[CAM] reopen failed, retrying..." << std::endl;
//        if (!reconnectTimer) reconnectTimer = new QTimer(this);
//        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
//    }
//}















////#include "camera.h"

////#include <QFile>
////#include <QTimer>

////#include <opencv2/imgproc.hpp>
////#include <opencv2/core/utility.hpp>

////#include <iostream>
////#include <chrono>
////#include <thread>

////// ============================================================
////// USB helper
////// ============================================================
////static void tuneCaptureUSB(cv::VideoCapture &cap, int width, int height, int fps)
////{
////    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
////    cap.set(cv::CAP_PROP_FPS, fps);
////    cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
////    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

////#ifdef CV_CAP_PROP_BUFFERSIZE
////    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
////#endif
////}

////// ============================================================
////// CSI helper
//////
////// ВАЖНО:
////// 1) Не используем RGBA -> потом cvtColor.
////// 2) Сразу приводим к BGR через BGRx + videoconvert.
////// 3) Для стабильности по умолчанию работаем на 30 fps.
////// ============================================================
////static std::string buildCsiPipeline(int sensorId, int width, int height, int fps, int flipMethod)
////{
////    return cv::format(
////        "nvarguscamerasrc sensor-id=%d ! "
////        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1, format=NV12 ! "
////        "queue leaky=downstream max-size-buffers=1 ! "
////        "nvvidconv flip-method=%d ! "
////        "video/x-raw, format=BGRx ! "
////        "videoconvert ! "
////        "video/x-raw, format=BGR ! "
////        "appsink drop=true max-buffers=1 sync=false",
////        sensorId, width, height, fps, flipMethod
////    );
////}

////Camera::Camera([[maybe_unused]] QObject *parent, int width)
////{
////    CV_Assert(width > 100);

////    setStopped(true);
////    setStarted(false);

////    camera_width = width;

////    // По умолчанию этот файл заточен под CSI IMX477.
////    // Если нужен USB — вызови setSource(Source::USB_V4L2) до запуска.
////    camera_source = Source::CSI_ARGUS;
////    applySourceDefaults();

////    Yolo = new my_yolo;
////    Yolo->setPerfPrintEvery(30);
////}

////Camera::~Camera()
////{
////    closeDevice();
////    delete Yolo;
////}

////void Camera::setSource(Source src)
////{
////    camera_source = src;
////    applySourceDefaults();
////}

////void Camera::setRequestedWidth(int width)
////{
////    camera_width = width;
////    applySourceDefaults();
////}

////void Camera::applySourceDefaults()
////{
////    if (camera_source == Source::CSI_ARGUS)
////    {
////        // Для CSI IMX477 держим только стабильные режимы
////        // 1920x1080@30 или 3840x2160@30
////        if (camera_width >= 3840) {
////            camera_width  = 3840;
////            camera_height = 2160;
////            camera_fps    = 20;
////        } else {
////            camera_width  = 3840;
////            camera_height = 2160;
////            camera_fps    = 30;   // специально 30, не 60
////        }
////    }
////    else
////    {
////        // Старое поведение для USB
////        if (camera_width == 1920) {
////            camera_height = 1080;
////            camera_fps    = 60;
////        } else if (camera_width == 3840) {
////            camera_height = 2160;
////            camera_fps    = 30;
////        } else if (camera_width == 4032) {
////            camera_height = 3040;
////            camera_fps    = 20;
////        } else {
////            camera_width  = 1920;
////            camera_height = 1080;
////            camera_fps    = 30;
////        }
////    }
////}

////bool Camera::openUsbDevice()
////{
////    const std::string devPath = "/dev/video0";
////    std::cout << "[CAM][USB] try open: " << devPath << std::endl;

////    if (!video.open(devPath, cv::CAP_V4L2)) {
////        std::cerr << "[CAM][USB] open failed: " << devPath << std::endl;
////        return false;
////    }

////    tuneCaptureUSB(video, camera_width, camera_height, camera_fps);

////    cv::Mat test;
////    for (int i = 0; i < 10; ++i) {
////        if (video.read(test) && !test.empty()) {
////            std::cout << "[CAM][USB] opened OK: " << devPath
////                      << " (" << test.cols << "x" << test.rows << ")" << std::endl;
////            return true;
////        }
////        std::this_thread::sleep_for(std::chrono::milliseconds(50));
////    }

////    std::cerr << "[CAM][USB] open ok, but no frames." << std::endl;
////    video.release();
////    return false;
////}

////bool Camera::openCsiDevice()
////{
////    const std::string pipeline = buildCsiPipeline(
////        csi_sensor_id,
////        camera_width,
////        camera_height,
////        camera_fps,
////        csi_flip_method
////    );

////    std::cout << "[CAM][CSI] try open pipeline:\n" << pipeline << std::endl;

////    if (!video.open(pipeline, cv::CAP_GSTREAMER)) {
////        std::cerr << "[CAM][CSI] open failed (pipeline)" << std::endl;
////        return false;
////    }

////#ifdef CV_CAP_PROP_BUFFERSIZE
////    video.set(cv::CAP_PROP_BUFFERSIZE, 1);
////#endif

////    // Даем Argus немного прогреться
////    std::this_thread::sleep_for(std::chrono::milliseconds(300));

////    cv::Mat test;
////    for (int i = 0; i < 20; ++i)
////    {
////        if (video.read(test) && !test.empty()) {
////            std::cout << "[CAM][CSI] opened OK: ("
////                      << test.cols << "x" << test.rows
////                      << "), channels=" << test.channels() << std::endl;
////            return true;
////        }
////        std::this_thread::sleep_for(std::chrono::milliseconds(80));
////    }

////    std::cerr << "[CAM][CSI] open ok, but no frames." << std::endl;
////    video.release();

////    // ВАЖНО: после release дать Argus отдышаться
////    std::this_thread::sleep_for(std::chrono::milliseconds(500));
////    return false;
////}

////bool Camera::openDevice()
////{
////    return (camera_source == Source::CSI_ARGUS) ? openCsiDevice()
////                                                : openUsbDevice();
////}

////void Camera::closeDevice()
////{
////    if (video.isOpened()) {
////        video.release();
////        std::cout << "[CAM] device closed" << std::endl;

////        // Для CSI это важно: не открывать снова мгновенно
////        if (camera_source == Source::CSI_ARGUS) {
////            std::this_thread::sleep_for(std::chrono::milliseconds(500));
////        }
////    }
////}

////void Camera::run()
////{
////    setStopped(false);
////    setStarted(true);

////    // Для CSI не надо агрессивно долбить reopen.
////    // Несколько попыток при старте допустимы.
////    const int open_attempts = (camera_source == Source::CSI_ARGUS) ? 5 : 20;

////    for (int attempt = 0; attempt < open_attempts && !openDevice(); ++attempt) {
////        std::this_thread::sleep_for(std::chrono::milliseconds(300));
////    }

////    if (!video.isOpened()) {
////        std::cerr << "[CAM] start failed: device not opened" << std::endl;
////        setStopped(true);
////        setStarted(false);
////        return;
////    }

////    if (Yolo) {
////        Yolo->setPerfDivisor(infer_every_n);
////    }

////    const cv::Size out_size(1280, 720);
////    const int perf_every = 60;

////    double sum_read = 0.0;
////    double sum_yolo = 0.0;
////    double sum_vis  = 0.0;
////    double sum_emit = 0.0;

////    int perf_frames = 0;
////    int perf_yolo_runs = 0;

////    double fps_ema = 0.0;
////    const double fps_alpha = 0.15;
////    int64 t_prev = cv::getTickCount();

////    int bad_reads_in_row = 0;
////    const int BAD_READS_LIMIT = 5;

////    int frame_id = 0;

////    while (!isStopped())
////    {
////        cv::TickMeter tm_read, tm_yolo, tm_vis, tm_emit;
////        cv::Mat frame;

////        // ========================================================
////        // 1) read// ========================================================
////        tm_read.start();
////        const bool ok = video.read(frame);
////        tm_read.stop();

////        if (!ok || frame.empty())
////        {
////            if (camera_source == Source::CSI_ARGUS)
////            {
////                // Для CSI НИКАКОГО агрессивного reopen внутри цикла.
////                // Иначе Argus очень легко уходит в плохое состояние.
////                std::cerr << "[CAM][CSI] read failed or empty frame. Stop stream." << std::endl;
////                break;
////            }

////            // USB ветка: мягкий reconnect допустим
////            bad_reads_in_row++;
////            if (bad_reads_in_row >= BAD_READS_LIMIT)
////            {
////                std::cerr << "[CAM][USB] stream lost. Reopening..." << std::endl;
////                closeDevice();
////                std::this_thread::sleep_for(std::chrono::milliseconds(200));

////                bool reopened = false;
////                for (int t = 0; t < 30 && !isStopped(); ++t) {
////                    if (openDevice()) {
////                        reopened = true;
////                        break;
////                    }
////                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
////                }

////                if (reopened) {
////                    std::cerr << "[CAM][USB] reconnected OK\n";
////                    bad_reads_in_row = 0;
////                    t_prev = cv::getTickCount();
////                } else {
////                    std::cerr << "[CAM][USB] reconnect failed\n";
////                    break;
////                }
////                continue;
////            }

////            std::this_thread::sleep_for(std::chrono::milliseconds(10));
////            continue;
////        }

////        bad_reads_in_row = 0;

////        // ========================================================
////        // 2) Привести кадр к BGR, если вдруг не BGR
////        // Для CSI после нашего pipeline он уже должен быть BGR.
////        // ========================================================
////        if (frame.channels() == 4) {
////            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
////        } else if (frame.channels() == 1) {
////            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
////        }

////        // ========================================================
////        // 3) FPS
////        // ========================================================
////        int64 t_now = cv::getTickCount();
////        const double dt = static_cast<double>(t_now - t_prev) / cv::getTickFrequency();
////        t_prev = t_now;

////        const double fps_inst = (dt > 1e-9) ? (1.0 / dt) : 0.0;
////        if (fps_ema <= 0.0) fps_ema = fps_inst;
////        else fps_ema = fps_alpha * fps_inst + (1.0 - fps_alpha) * fps_ema;

////        // ========================================================
////        // 4) YOLO (можно отключить через setYoloEnabled(false))
////        // ========================================================
////        const bool do_infer = yolo_enabled &&
////                              (infer_every_n <= 1 || ((frame_id % infer_every_n) == 0));

////        if (do_infer && Yolo)
////        {
////            tm_yolo.start();
////            (void)Yolo->run(frame);
////            tm_yolo.stop();
////            perf_yolo_runs++;
////        }

////        // ========================================================
////        // 5) vis
////        // ========================================================
////        tm_vis.start();

////        cv::resize(frame, frame, out_size);

////        char buf[160];
////        std::snprintf(buf, sizeof(buf),
////                      "FPS: %.1f  infer: %s",
////                      fps_ema,
////                      yolo_enabled ? ("1/" + std::to_string(infer_every_n)).c_str() : "OFF");

////        cv::putText(frame, buf, cv::Point(15, 30),
////                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

////        tm_vis.stop();

////        // ========================================================
////        // 6) emit
////        // ========================================================
////        tm_emit.start();
////        emit output_frame(frame);
////        tm_emit.stop();

////        // ========================================================
////        // PERF
////        // ========================================================
////        sum_read += tm_read.getTimeMilli();
////        sum_yolo += tm_yolo.getTimeMilli();
////        sum_vis  += tm_vis.getTimeMilli();
////        sum_emit += tm_emit.getTimeMilli();
////        perf_frames++;

////        if (perf_frames >= perf_every)
////        {
////            const double avg_read = sum_read / perf_frames;
////            const double avg_yolo = sum_yolo / perf_frames;
////            const double avg_vis  = sum_vis  / perf_frames;
////            const double avg_emit = sum_emit / perf_frames;

////            std::cerr
////                << "[PERF CAM] "
////                << "1.read=" << avg_read << "ms, "
////                << "2.yolo(avg)=" << avg_yolo << "ms, "
////                << "3.vis=" << avg_vis << "ms, "
////                << "4.emit=" << avg_emit << "ms, "
////                << "infer_runs=" << perf_yolo_runs << "/" << perf_frames
////                << "\n";

////            sum_read = sum_yolo = sum_vis = sum_emit = 0.0;
////            perf_frames = 0;
////            perf_yolo_runs = 0;
////        }

////        frame_id++;
////    }

////    closeDevice();
////    setStopped(true);
////    setStarted(false);
////}

////void Camera::checkAndReconnect()
////{
////    if (isStopped()) {
////        return;
////    }

////    // ========================================================
////    // Для CSI reconnect через этот slot ОТКЛЮЧЁН СПЕЦИАЛЬНО.
////    // Иначе легко получить:
////    // - битые кадры
////    // - полосы
////    // - зависание Argus
////    // ========================================================
////    if (camera_source == Source::CSI_ARGUS) {
////        return;
////    }

////    const std::string devPath = "/dev/video0";
////    if (!QFile::exists(QString::fromStdString(devPath))) {
////        std::cerr << "[CAM][USB] device missing, waiting..." << std::endl;
////        if (!reconnectTimer) reconnectTimer = new QTimer(this);
////        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
////        return;
////    }

////    if (video.isOpened()) {
////        cv::Mat test;
////        if (!video.read(test) || test.empty()) {
////            std::cerr << "[CAM][USB] stream lost, reopening..." << std::endl;
////            video.release();
////        } else {
////            if (!reconnectTimer) reconnectTimer = new QTimer(this);
////            reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
////            return;
////        }
////    }

////    if (openDevice()) {
////        std::cout << "[CAM][USB] reconnected successfully!" << std::endl;
////        if (!reconnectTimer) reconnectTimer = new QTimer(this);
////        reconnectTimer->singleShot(2000, this, &Camera::checkAndReconnect);
////    } else {
////        std::cerr << "[CAM][USB] reopen failed, retrying..." << std::endl;
////        if (!reconnectTimer) reconnectTimer = new QTimer(this);
////        reconnectTimer->singleShot(1000, this, &Camera::checkAndReconnect);
////    }
////}
