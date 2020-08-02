
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ExtractContiguousTextRegions.h"

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
__device__ __forceinline__ T read_pixel(T const* const grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, std::int32_t width, std::int32_t height)
{
	return *(grayscale + width * height * 3 * batch + width * height * channel + width * y + x);
}

union load_16bytes
{
	uint4 u128;
	struct
	{
		std::uint8_t u8s[16];
	};
};

// copy extracted frames to contiguous storage space
__global__ void ExtractContiguousTextRegions_kernel(
	std::uint8_t const* const in_frames,
	std::uint8_t* out_frames,
	std::int32_t const* const in_source_index,
	std::int32_t frame_height,
	std::int32_t reduced_frame_width
)
{
	load_16bytes const* const in_16bytes(reinterpret_cast<load_16bytes const* const>(in_frames));
	load_16bytes* out_16bytes(reinterpret_cast<load_16bytes*>(out_frames));
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	if (ix >= reduced_frame_width || iy >= frame_height / 16)
		return;
	std::int32_t source_frame(in_source_index[iz]);
	if (source_frame == -1)
	{
		load_16bytes zeros{};
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, zeros, zeros, zeros, reduced_frame_width, frame_height);
		}
	}
	else
	{
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			auto r(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 0, reduced_frame_width, frame_height));
			auto g(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 1, reduced_frame_width, frame_height));
			auto b(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 2, reduced_frame_width, frame_height));
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, r, g, b, reduced_frame_width, frame_height);
		}
	}
}

void ExtractContiguousTextRegions(
	cudawrapper::CUDADeviceMemoryUnique<uchar> const& in_all_regions,
	std::vector<std::int32_t> in_region_indices,
	cudawrapper::CUDADeviceMemoryUnique<uchar>& out_contiguous_regions,
	std::int32_t& out_batches,
	cudawrapper::CUDADeviceMemoryUnique<std::int32_t>& tmp_region_indices_gpu,
	std::int32_t region_width,
	std::int32_t region_height,
	std::size_t output_batch_size,
	CUstream stream
)
{
	auto reduced_region_width(region_width / 16);
	auto num_contiguous_regions(in_region_indices.size());

	if (num_contiguous_regions % output_batch_size != 0)
	{
		auto pad_size(output_batch_size - (num_contiguous_regions % output_batch_size));
		num_contiguous_regions += pad_size;
		for (std::size_t i(0); i < pad_size; ++i)
			in_region_indices.emplace_back(-1);
	}

	out_batches = num_contiguous_regions / output_batch_size;

	if (out_contiguous_regions.empty() || out_contiguous_regions.size() < num_contiguous_regions * 3 * region_height * region_width)
		out_contiguous_regions.reallocate(num_contiguous_regions * 3 * region_height * region_width);
	if (tmp_region_indices_gpu.empty() || tmp_region_indices_gpu.size() < num_contiguous_regions)
		tmp_region_indices_gpu.reallocate(num_contiguous_regions);

	tmp_region_indices_gpu.upload(in_region_indices, stream);

	dim3 block(32, 32, 1);
	dim3 grid((reduced_region_width - 1) / 32 + 1, (region_height / 16 - 1) / 32 + 1, num_contiguous_regions);

	// copy
	ExtractContiguousTextRegions_kernel<<<grid, block, 0, stream>>>(in_all_regions, out_contiguous_regions, tmp_region_indices_gpu, region_height, reduced_region_width);
	ck2(cudaGetLastError());
}
