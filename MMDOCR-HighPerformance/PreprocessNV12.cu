
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "PreprocessNV12.h"

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

__device__ __forceinline__ float3 read_pixel(float* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height)
{
	float r, g, b;
	r = *(rgb + width * height * 3 * batch + width * height * 0 + width * y + x);
	g = *(rgb + width * height * 3 * batch + width * height * 1 + width * y + x);
	b = *(rgb + width * height * 3 * batch + width * height * 2 + width * y + x);
	return float3{ r, g, b };
}

__device__ __forceinline__ uchar3 read_pixel(std::uint8_t* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height)
{
	std::uint8_t r, g, b;
	r = *(rgb + width * height * 3 * batch + width * height * 0 + width * y + x);
	g = *(rgb + width * height * 3 * batch + width * height * 1 + width * y + x);
	b = *(rgb + width * height * 3 * batch + width * height * 2 + width * y + x);
	return uchar3{ r, g, b };
}

__device__ __forceinline__ std::uint8_t saturate_u8(float v)
{
	return static_cast<std::uint8_t>(max(min(v, 255.0f), 0.0f));
}

__device__ __forceinline__ uchar3 yuv2rgb(std::uint8_t y_, std::uint8_t u_, std::uint8_t v_)
{
	float luma(static_cast<float>(y_));
	float u(static_cast<float>(u_));
	float v(static_cast<float>(v_));
	return uchar3{
		saturate_u8(luma + (1.40200f * (v - 128.0f))),
		saturate_u8(luma - (0.34414f * (u - 128.0f)) - (0.71414f * (v - 128.0f))),
		saturate_u8(luma + (1.77200f * (u - 128.0f))),
	};
}

__device__ __forceinline__ float l1_dist(float3 const& a, float3 const& b)
{
	return std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z);
}

__device__ __forceinline__ float sqr(float const a)
{
	return a * a;
}

__device__ __forceinline__ float3 operator*(float const& lhs, float3 const& rhs)
{
	return float3{ lhs * rhs.x, lhs * rhs.y, lhs * rhs.z };
}

__device__ __forceinline__ float3 operator/(float3 const& lhs, float const& rhs)
{
	return float3{ lhs.x / rhs, lhs.y / rhs, lhs.z / rhs };
}

__device__ __forceinline__ float3 operator+(float3 const& lhs, float3 const& rhs)
{
	return float3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

// convert from NV12 to RGB format and do a nearest resampling
__global__ void PreprocessNV12_stage_1_kernel(
	std::uint8_t const* const nv12,
	std::uint8_t* tmp,
	std::int32_t const input_width,
	std::int32_t const input_height,
	std::int32_t const input_frame_size,
	std::int32_t const target_width,
	std::int32_t const target_height,
	std::int32_t const output_width,
	std::int32_t const output_height
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);

	if (ix >= target_width || iy >= target_height)
	{
		write_pixel(tmp, iz, ix, iy, (std::uint8_t)0, (std::uint8_t)0, (std::uint8_t)0, output_width, output_height);
		return;
	}
	// step 1: use border color
	std::int32_t reflected_x(min(ix, target_width - 1));
	std::int32_t reflected_y(min(iy, target_height - 1));
	//std::int32_t reflected_x(target_width - std::abs(ix - target_width + 1) - 1);
	//std::int32_t reflected_y(target_height - std::abs(iy - target_height + 1) - 1);
	// step 2: get sampling location in original image coordinate
	float2 uv{ __int2float_rz(reflected_x) / __int2float_rz(target_width - 1), __int2float_rz(reflected_y) / __int2float_rz(target_height - 1) };
	int2 pos{ __float2int_rz(uv.x * __int2float_rz(input_width - 1) + 0.5f),  __float2int_rz(uv.y * __int2float_rz(input_height - 1) + 0.5f) };
	int2 pos_half{ __float2int_rz(uv.x * __int2float_rz(input_width / 2 - 1) + 0.5f),  __float2int_rz(uv.y * __int2float_rz(input_height / 2 - 1) + 0.5f) };
	// step 3: get YUV color from NV12 layout
	std::uint8_t color_y(*(nv12 + input_frame_size * iz + pos.y * input_width + pos.x));
	auto chroma_base(reinterpret_cast<uchar2 const* const>(nv12 + input_frame_size * iz + input_width * input_height));
	auto color_uv(chroma_base[input_width / 2 * pos_half.y + pos_half.x]);
	// step 4: convert to RGB bytes
	auto rgb(yuv2rgb(color_y, color_uv.x, color_uv.y));
	// step 5: store in tmp array
	write_pixel(tmp, iz, ix, iy, rgb.x, rgb.y, rgb.z, output_width, output_height);
	// step 6: sync
}

// bilateral filter
template <std::int32_t bilateral_ks = 11>
__global__ void PreprocessNV12_stage_2_kernel(
	std::uint8_t* tmp,
	std::uint8_t* rgb,
	std::int32_t const output_width,
	std::int32_t const output_height,
	float const sigma_spatial2_inv_half,
	float const sigma_color2_inv_half
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	auto center(read_pixel(tmp, iz, ix, iy, output_width, output_height));
	// step 7: load into shared memory
	__shared__ uchar3 tile[32 + bilateral_ks - 1][32 + bilateral_ks - 1];
	std::int32_t const half_ks(bilateral_ks >> 1);
	std::int32_t sx(threadIdx.x), sy(threadIdx.y);
	// FIXME: the following load code contains WAW hazard
	// load top left
	tile[sx][sy] = read_pixel(tmp, iz, std::abs(ix - half_ks), std::abs(iy - half_ks), output_width, output_height); // reflect101
	// load top right
	tile[sx + bilateral_ks - 1][sy] = read_pixel(tmp, iz, output_width - std::abs(ix + half_ks - output_width + 1) - 1, std::abs(iy - half_ks), output_width, output_height); // reflect101
	// load bottom left
	tile[sx][sy + bilateral_ks - 1] = read_pixel(tmp, iz, std::abs(ix - half_ks), output_height - std::abs(iy + half_ks - output_height + 1) - 1, output_width, output_height); // reflect101
	// load bottom right
	tile[sx + bilateral_ks - 1][sy + bilateral_ks - 1] = read_pixel(tmp, iz, output_width - std::abs(ix + half_ks - output_width + 1) - 1, output_height - std::abs(iy + half_ks - output_height + 1) - 1, output_width, output_height); // reflect101
	__syncthreads();
	// step 8: bilateral filter in shared memory
	float3 sum_value{ 0, 0, 0 };
	float r2(half_ks * half_ks);
	float3 center_fp32{ static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z) };
	float sum_weight(0.0f);
	for (std::int32_t x(-half_ks); x <= half_ks; ++x)
		for (std::int32_t y(-half_ks); y <= half_ks; ++y)
		{
			float dist2(x * x + y * y);
			if (dist2 > r2)
				continue;
			uchar3 const& cur(tile[x + half_ks + sx][y + half_ks + sy]);
			float3 cur_fp32{ static_cast<float>(cur.x), static_cast<float>(cur.y), static_cast<float>(cur.z) };
			float weight = std::exp(dist2 * sigma_spatial2_inv_half + sqr(l1_dist(cur_fp32, center_fp32)) * sigma_color2_inv_half);
			sum_value = sum_value + weight * cur_fp32;
			sum_weight += weight;
		}
	// step 9: store
	float3 final_value(sum_value / sum_weight);
	write_pixel(rgb, iz, ix, iy, static_cast<std::uint8_t>(final_value.x), static_cast<std::uint8_t>(final_value.y), static_cast<std::uint8_t>(final_value.z), output_width, output_height);
}

// (batch) from raw decoded yuv(batch first) to resized, padded and bilateral filtered RGB image of type uint8 range [0, 255] (NCHW format)
void PreprocessNV12(
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& output,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& output_raw,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> const& batched_yuv_frames,
	std::size_t batch_size,
	std::size_t nv12_frame_size_in_bytes,
	std::int32_t input_width,
	std::int32_t input_height,
	std::int32_t canvas_size,
	std::int32_t* out_output_width,
	std::int32_t* out_output_height,
	CUstream stream
)
{
	// calculate output shape
	float ratio(static_cast<float>(canvas_size) / std::max(input_width, input_height));
	std::int32_t target_width(static_cast<std::int32_t>(std::round(ratio * input_width))), target_height(static_cast<std::int32_t>(std::round(ratio * input_height)));
	std::int32_t output_width(target_width % 32 == 0 ? target_width : (target_width + (32 - (target_width % 32))));
	std::int32_t output_height(target_height % 32 == 0 ? target_height : (target_height + (32 - (target_height % 32))));

	if (out_output_width)
		*out_output_width = output_width;
	if (out_output_height)
		*out_output_height = output_height;

	// allocate ouput buffer
	if (output.empty() || output.size() != batch_size * 3 * output_width * output_height)
		output.reallocate(batch_size * 3 * output_width * output_height);

	// if not initialized or batch_size changed
	if (output_raw.empty() || output_raw.size() != batch_size * 3 * output_width * output_height)
		// allocate new tmp buffer
		output_raw.reallocate(batch_size * 3 * output_width * output_height);

	dim3 block(32, 32, 1);
	dim3 grid(output_width / 32, output_height / 32, batch_size);

	float const sigma_spatial = 80.0f;
	float const sigma_color = 80.0f;

	float const sigma_spatial2_inv_half = -0.5f / (sigma_spatial * sigma_spatial);
	float const sigma_color2_inv_half = -0.5f / (sigma_color * sigma_color);

	PreprocessNV12_stage_1_kernel<<<grid, block, 0, stream>>>(
		batched_yuv_frames,
		output_raw,
		input_width,
		input_height,
		nv12_frame_size_in_bytes,
		target_width,
		target_height,
		output_width,
		output_height
		);

	ck2(cudaGetLastError());

	PreprocessNV12_stage_2_kernel<<<grid, block, 0, stream>>>(
		output_raw,
		output,
		output_width,
		output_height,
		sigma_spatial2_inv_half,
		sigma_color2_inv_half
		);

	ck2(cudaGetLastError());
}
