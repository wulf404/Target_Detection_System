#include "cuda_preprocess.h"
#include <cuda_runtime.h>

static __global__ void k_bgr_u8_to_nchw_f32(
    const uchar3* __restrict__ src, int src_step_bytes,
    float* __restrict__ dst, int out_w, int out_h,
    float scale, float mean_b, float mean_g, float mean_r,
    int swapRB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    const uchar3* row = (const uchar3*)((const unsigned char*)src + y * src_step_bytes);
    uchar3 p = row[x]; // BGR

    float b = ((float)p.x - mean_b) * scale;
    float g = ((float)p.y - mean_g) * scale;
    float r = ((float)p.z - mean_r) * scale;


    int hw = out_h * out_w;
    int idx = y * out_w + x;

    if (swapRB) {
        dst[0 * hw + idx] = r;
        dst[1 * hw + idx] = g;
        dst[2 * hw + idx] = b;
    } else {
        dst[0 * hw + idx] = b;
        dst[1 * hw + idx] = g;
        dst[2 * hw + idx] = r;
    }
}

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
    cudaStream_t stream)
{
    if (resized_bgr_u8.empty()) return false;
    if (resized_bgr_u8.type() != CV_8UC3) return false;
    if (resized_bgr_u8.cols != out_w || resized_bgr_u8.rows != out_h) return false;

    out_blob_f32.create(out_h * 3, out_w, CV_32F);

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    const uchar3* src_ptr = (const uchar3*)resized_bgr_u8.ptr<unsigned char>();
    float* dst_ptr = (float*)out_blob_f32.ptr<float>();

    k_bgr_u8_to_nchw_f32<<<grid, block, 0, stream>>>(
        src_ptr,
        (int)resized_bgr_u8.step,
        dst_ptr,
        out_w, out_h,
        scale, mean_b, mean_g, mean_r,
        swapRB ? 1 : 0
    );

    return (cudaGetLastError() == cudaSuccess);
}

bool cuda_preprocess_bgr_to_nchw_ptr(
    const cv::cuda::GpuMat& resized_bgr_u8,
    float* out_blob_f32,
    int out_w,
    int out_h,
    float scale,
    float mean_b,
    float mean_g,
    float mean_r,
    bool swapRB,
    cudaStream_t stream)
{
    if (resized_bgr_u8.empty()) return false;
    if (resized_bgr_u8.type() != CV_8UC3) return false;
    if (resized_bgr_u8.cols != out_w || resized_bgr_u8.rows != out_h) return false;
    if (!out_blob_f32) return false;

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    const uchar3* src_ptr = (const uchar3*)resized_bgr_u8.ptr<unsigned char>();

    k_bgr_u8_to_nchw_f32<<<grid, block, 0, stream>>>(
        src_ptr,
        (int)resized_bgr_u8.step,
        out_blob_f32,
        out_w, out_h,
        scale, mean_b, mean_g, mean_r,
        swapRB ? 1 : 0
    );

    return (cudaGetLastError() == cudaSuccess);
}
