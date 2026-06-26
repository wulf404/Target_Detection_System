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

static __device__ __forceinline__ uchar3 read_bgr_pixel(
    const uchar3* __restrict__ src,
    int src_step_bytes,
    int x,
    int y)
{
    const uchar3* row = (const uchar3*)((const unsigned char*)src + y * src_step_bytes);
    return row[x];
}

static __global__ void k_bgr_resize_to_nchw_f32(
    const uchar3* __restrict__ src, int src_step_bytes,
    int src_w, int src_h,
    float* __restrict__ dst, int out_w, int out_h,
    float scale, float mean_b, float mean_g, float mean_r,
    int swapRB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    const float src_x = ((float)x + 0.5f) * ((float)src_w / (float)out_w) - 0.5f;
    const float src_y = ((float)y + 0.5f) * ((float)src_h / (float)out_h) - 0.5f;

    const float fx = fminf(fmaxf(src_x, 0.0f), (float)(src_w - 1));
    const float fy = fminf(fmaxf(src_y, 0.0f), (float)(src_h - 1));

    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const int x1 = (x0 + 1 < src_w) ? x0 + 1 : src_w - 1;
    const int y1 = (y0 + 1 < src_h) ? y0 + 1 : src_h - 1;
    const float wx = fx - (float)x0;
    const float wy = fy - (float)y0;

    const uchar3 p00 = read_bgr_pixel(src, src_step_bytes, x0, y0);
    const uchar3 p01 = read_bgr_pixel(src, src_step_bytes, x1, y0);
    const uchar3 p10 = read_bgr_pixel(src, src_step_bytes, x0, y1);
    const uchar3 p11 = read_bgr_pixel(src, src_step_bytes, x1, y1);

    const float w00 = (1.0f - wx) * (1.0f - wy);
    const float w01 = wx * (1.0f - wy);
    const float w10 = (1.0f - wx) * wy;
    const float w11 = wx * wy;

    const float b_u8 = w00 * (float)p00.x + w01 * (float)p01.x +
                       w10 * (float)p10.x + w11 * (float)p11.x;
    const float g_u8 = w00 * (float)p00.y + w01 * (float)p01.y +
                       w10 * (float)p10.y + w11 * (float)p11.y;
    const float r_u8 = w00 * (float)p00.z + w01 * (float)p01.z +
                       w10 * (float)p10.z + w11 * (float)p11.z;

    const float b = (b_u8 - mean_b) * scale;
    const float g = (g_u8 - mean_g) * scale;
    const float r = (r_u8 - mean_r) * scale;

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

bool cuda_preprocess_bgr_resize_to_nchw_ptr(
    const cv::cuda::GpuMat& src_bgr_u8,
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
    if (src_bgr_u8.empty()) return false;
    if (src_bgr_u8.type() != CV_8UC3) return false;
    if (src_bgr_u8.cols <= 0 || src_bgr_u8.rows <= 0) return false;
    if (out_w <= 0 || out_h <= 0) return false;
    if (!out_blob_f32) return false;

    if (src_bgr_u8.cols == out_w && src_bgr_u8.rows == out_h) {
        return cuda_preprocess_bgr_to_nchw_ptr(
            src_bgr_u8,
            out_blob_f32,
            out_w,
            out_h,
            scale,
            mean_b,
            mean_g,
            mean_r,
            swapRB,
            stream
        );
    }

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    const uchar3* src_ptr = (const uchar3*)src_bgr_u8.ptr<unsigned char>();

    k_bgr_resize_to_nchw_f32<<<grid, block, 0, stream>>>(
        src_ptr,
        (int)src_bgr_u8.step,
        src_bgr_u8.cols,
        src_bgr_u8.rows,
        out_blob_f32,
        out_w,
        out_h,
        scale,
        mean_b,
        mean_g,
        mean_r,
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
