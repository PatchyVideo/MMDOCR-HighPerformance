
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "RgbBytesToRgbFp32.h"

template<typename T>
__device__ __forceinline__ void write_pixel_gs(T* grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, T val, std::int32_t width, std::int32_t height)
{
	*(grayscale + width * height * batch + width * y + x) = val;
}

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, T val, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * channel + width * y + x) = val;
}

template<typename T>
__device__ __forceinline__ T read_pixel(T const* const grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, std::int32_t width, std::int32_t height)
{
	return *(grayscale + width * height * 3 * batch + width * height * channel + width * y + x);
}

union load_4bytes
{
	std::uint8_t u8s[4];
};

union store_4fp32s
{
	float fp32s[4];
};

__global__ void RgbBytesToRgbFp32_kernel(
	std::uint8_t const* const in_u8,
	float* out_fp32,
	std::int32_t reduced_frame_width,
	std::int32_t height
)
{
	load_4bytes const* const in_4bytes(reinterpret_cast<load_4bytes const* const>(in_u8));
	store_4fp32s* out_4fp32s(reinterpret_cast<store_4fp32s*>(out_fp32));
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	if (ix >= reduced_frame_width || iy >= height / 4)
		return;
	for (int i(0); i < 4; ++i)
	{
		store_4fp32s out;
		auto r(read_pixel(in_4bytes, iz, ix, iy * 4 + i, 0, reduced_frame_width, height));
#pragma unroll
		for (int j(0); j < 4; ++j)
		{
			out.fp32s[j] = static_cast<float>(r.u8s[j]) / 127.5f - 1.0f;
		}
		write_pixel(out_4fp32s, iz, ix, iy * 4 + i, 0, out, reduced_frame_width, height);
		auto g(read_pixel(in_4bytes, iz, ix, iy * 4 + i, 1, reduced_frame_width, height));
#pragma unroll
		for (int j(0); j < 4; ++j)
		{
			out.fp32s[j] = static_cast<float>(g.u8s[j]) / 127.5f - 1.0f;
		}
		write_pixel(out_4fp32s, iz, ix, iy * 4 + i, 1, out, reduced_frame_width, height);
		auto b(read_pixel(in_4bytes, iz, ix, iy * 4 + i, 2, reduced_frame_width, height));
#pragma unroll
		for (int j(0); j < 4; ++j)
		{
			out.fp32s[j] = static_cast<float>(b.u8s[j]) / 127.5f - 1.0f;
		}
		write_pixel(out_4fp32s, iz, ix, iy * 4 + i, 2, out, reduced_frame_width, height);
	}
}

void RgbBytesToRgbFp32(
	std::uint8_t const * const input_u8_frames,
	cudawrapper::CUDADeviceMemoryUnique<float>& out_fp32_frames,
	std::size_t input_frame_count,
	std::int32_t width,
	std::int32_t height,
	CUstream stream
)
{
	if (out_fp32_frames.empty() || out_fp32_frames.size() < input_frame_count * 3 * width * height)
		out_fp32_frames.reallocate(input_frame_count * 3 * width * height);

	auto reduced_frame_width(width / 4);

	dim3 block(32, 32, 1);
	dim3 grid((reduced_frame_width - 1) / 32 + 1, (height / 4 - 1) / 32 + 1, input_frame_count);

	RgbBytesToRgbFp32_kernel<<<grid, block, 0, stream>>>(input_u8_frames, out_fp32_frames, reduced_frame_width, height);
	ck2(cudaGetLastError());
}
