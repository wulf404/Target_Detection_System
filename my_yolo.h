#ifndef MY_YOLO_H
#define MY_YOLO_H

#include <common_constants.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <NvInfer.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <memory>

#if ENABLE_CUDA_PREPROCESS
#include "cuda_preprocess.h"
#endif

class my_yolo
{
public:
    my_yolo();
    ~my_yolo();

    // frame BGR CV_8UC3, рисование боксов может быть внутри (как сейчас)
    yolo_output run(cv::Mat &frame, double captureMs = 0.0);

    // печатать perf раз в N вызовов run()
    void setPerfPrintEvery(int n) { perf_print_every = (n <= 0 ? 1 : n); }

    // NEW: если инференс делаем раз в K кадров, делим тайминги на K,
    // чтобы в логе было “эквивалентно на кадр”
    void setPerfDivisor(int k) { perf_divisor = (k <= 0 ? 1 : k); }

private:
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* object) const
        {
            delete object;
        }
    };

    class TrtLogger final : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    struct TrtOutputBinding
    {
        std::string name;
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        nvinfer1::Dims dims{};
        void* device = nullptr;
        size_t bytes = 0;
        std::vector<float> host;
        std::vector<__half> hostHalf;
    };

    std::string weightPath;
    int yolo_width = 0, yolo_height = 0;

    TrtLogger trtLogger;
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> trtRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> trtEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> trtContext;
    std::string trtInputName;
    void* trtInputDevice = nullptr;
    size_t trtInputBytes = 0;
    std::vector<TrtOutputBinding> trtOutputs;
    cudaStream_t trtStream = nullptr;

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
    cv::Mat padded_inference_frame;
    cv::cuda::GpuMat gpu_frame_u8;
    cv::cuda::GpuMat gpu_resized_u8;
    cv::cuda::Stream gpu_stream;

    std::vector<cv::Mat> outs;

    // PERF
    int frame_counter = 0;
    int perf_print_every = 30;
    int perf_divisor = 1; // NEW

    cv::Point last_target_center;
    cv::Rect last_target_box;
    cv::Rect last_inference_roi;
    cv::Point2d track_center_f;
    cv::Point2d track_velocity_px_s;
    cv::Size2d track_box_size_f;
    uint64_t track_last_update_ms = 0;
    uint64_t track_last_detection_ms = 0;
    double track_quality = 0.0;
    int acquire_hit_frames = 0;
    int suspicious_frames = 0;
    bool have_last_target = false;
    bool have_track_model = false;
    bool track_confirmed = false;
    int lost_target_frames = 0;
    int roi_frames_since_full_scan = 0;

    void loadClasses(const std::string& namesPath);
    void initTensorRt();
    void releaseTensorRt();
    bool allocateTensorRtBuffers();
    bool runTensorRt(std::vector<cv::Mat>& outputMats);
};

#endif // MY_YOLO_H
