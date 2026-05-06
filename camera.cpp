#include "camera.h"
#include "app_config.h"

#include <QFile>
#include <QString>
#include <QTimer>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <cstdio>

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

    for (int i = 0; i < 10; ++i) {
        const std::string path = "/dev/video" + std::to_string(i);
        if (QFile::exists(QString::fromStdString(path))) {
            devices.push_back(path);
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
        publishDeviceState(true, devices.front());

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
                    publishDeviceState(true, current_device_path);
                    std::cout << "[CAM][USB][GST] opened OK: "
                              << current_device_path << " "
                              << test.cols << "x" << test.rows
                              << ", ch=" << test.channels()
                              << ", req=" << camera_width << "x" << camera_height
                              << "@" << camera_fps
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
    if (video.isOpened()) {
        video.release();
        std::cout << "[CAM] device closed" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    current_device_path.clear();
}

void Camera::run()
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
                (void)Yolo->run(frame);
                tm_yolo.stop();
                perf_yolo_runs++;
            }

            // ========================================================
            // 5) vis
            // ========================================================
            tm_vis.start();

            cv::resize(frame, frame, out_size);

            std::string infer_text = yolo_enabled
                ? ("1/" + std::to_string(infer_every_n))
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

void Camera::checkAndReconnect()
{
    if (isStopped()) {
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
