#ifndef CUDA_PREPROCESS_H
#define CUDA_PREPROCESS_H

#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>


bool cuda_preprocess_bgr_to_nchw(
    const cv::cuda::GpuMat& resized_bgr_u8,
    cv::cuda::GpuMat& out_blob_f32,
    int out_w,
    int out_h,
    float scale,
    float mean_b,
    float mean_g,
    float mean_r,
    bool swapRB,
    cudaStream_t stream
);

#endif // CUDA_PREPROCESS_H

