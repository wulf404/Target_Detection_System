#ifndef MY_YOLO_H
#define MY_YOLO_H

#include <common_constants.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

#if ENABLE_CUDA_PREPROCESS
#include "cuda_preprocess.h"
#endif

class my_yolo
{
public:
    my_yolo();

    // frame BGR CV_8UC3, рисование боксов может быть внутри (как сейчас)
    yolo_output run(cv::Mat &frame);

    // печатать perf раз в N вызовов run()
    void setPerfPrintEvery(int n) { perf_print_every = (n <= 0 ? 1 : n); }

    // NEW: если инференс делаем раз в K кадров, делим тайминги на K,
    // чтобы в логе было “эквивалентно на кадр”
    void setPerfDivisor(int k) { perf_divisor = (k <= 0 ? 1 : k); }

private:
    std::string weightPath;
    int yolo_width = 0, yolo_height = 0;

    cv::dnn::Net net;

    float confThreshold = 0.3f;
    float nmsThreshold  = 0.5f;

    double scale = 1.0/122.0;
    cv::Scalar mean = cv::Scalar(120,120,120);
    bool swapRB = true;

    std::vector<std::string> classes;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> keep_idx;

    yolo_output output;

    // GPU preprocess
    cv::cuda::GpuMat gpu_frame_u8;
    cv::cuda::GpuMat gpu_resized_u8;
    cv::cuda::GpuMat gpu_blob_f32;
    cv::cuda::Stream gpu_stream;

    std::vector<cv::Mat> outs;

    // PERF
    int frame_counter = 0;
    int perf_print_every = 30;
    int perf_divisor = 1; // NEW

    cv::Point last_target_center;
    bool have_last_target = false;
    int lost_target_frames = 0;

    void loadClasses(const std::string& namesPath);
};

#endif // MY_YOLO_H
