#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ExtractTextRegions.h"

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

__device__ __forceinline__ uchar3 read_pixel(std::uint8_t const* const rgb, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height)
{
	uchar r, g, b;
	r = *(rgb + width * height * 3 * batch + width * height * 0 + width * y + x);
	g = *(rgb + width * height * 3 * batch + width * height * 1 + width * y + x);
	b = *(rgb + width * height * 3 * batch + width * height * 2 + width * y + x);
	return uchar3{ r, g, b };
}

__global__ void ExtractTextRegions_kernel(
	std::uint8_t const* const frames,
	BBox const* const bboxes,
	subtitle_index const* const bbox_frame_map,
	std::uint8_t* regions,
	std::int32_t frame_width,
	std::int32_t frame_height,
	std::int32_t region_width, // multiple of 32
	std::int32_t region_height // 32
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	auto const& box(bboxes[iz]);
	auto ratio(__int2float_rz(box.width) / __int2float_rz(box.height));
	auto new_width(__float2int_rz(__int2float_rz(region_height) * ratio));
	auto src_frame_idx(bbox_frame_map[iz].first);
	if (new_width <= region_width)
	{
		// right pad zero
		if (ix >= new_width)
		{
			// set zero
			write_pixel(regions, iz, ix, iy, (std::uint8_t)0, (std::uint8_t)0, (std::uint8_t)0, region_width, region_height);
		}
		else
		{
			// nearest sample
			float2 uv{ __int2float_rz(ix) / __int2float_rz(new_width - 1), __int2float_rz(iy) / __int2float_rz(region_height - 1) };
			int2 pos{ __float2int_rz(uv.x * __int2float_rz(box.width - 1) + 0.5f), __float2int_rz(uv.y * __int2float_rz(box.height - 1) + 0.5f) };
			auto rgb(read_pixel(frames, src_frame_idx, pos.x + box.x, pos.y + box.y, frame_width, frame_height));
			write_pixel(regions, iz, ix, iy, rgb.x, rgb.y, rgb.z, region_width, region_height);
		}
	}
	else
	{
		// pad top and bottom zero
		auto new_height(__float2int_rz(__int2float_rz(region_width) / ratio));
		new_height += new_height & 1; // make multiple of 2
		auto pad_height((region_height - new_height) / 2);
		if (iy < pad_height || iy >= pad_height + new_height)
		{
			// set zero
			write_pixel(regions, iz, ix, iy, (std::uint8_t)0, (std::uint8_t)0, (std::uint8_t)0, region_width, region_height);
		}
		else
		{
			// nearest sample
			float2 uv{ __int2float_rz(ix) / __int2float_rz(region_width - 1), __int2float_rz(iy - pad_height) / __int2float_rz(new_height - 1) };
			int2 pos{ __float2int_rz(uv.x * __int2float_rz(box.width - 1) + 0.5f), __float2int_rz(uv.y * __int2float_rz(box.height - 1) + 0.5f) };
			auto rgb(read_pixel(frames, src_frame_idx, pos.x + box.x, pos.y + box.y, frame_width, frame_height));
			write_pixel(regions, iz, ix, iy, rgb.x, rgb.y, rgb.z, region_width, region_height);
		}
	}
}

void ExtractTextRegions(
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> const& in_frames,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& out_extracted_regions,
	cudawrapper::CUDADeviceMemoryUnique<BBox>& tmp_contiguous_bboxes_gpu,
	cudawrapper::CUDADeviceMemoryUnique<subtitle_index>& tmp_contiguous_individual_map_gpu,
	std::vector<BBox> const& bboxes_contiguous,
	std::vector<subtitle_index> contiguous_individual_map,
	std::int32_t width,
	std::int32_t height,
	std::int32_t text_region_height, // this is 32
	std::int32_t text_region_max_width,
	std::int32_t *out_text_region_width,
	CUstream stream
)
{
	assert(text_region_height == 32);
	std::size_t num_text_regions(bboxes_contiguous.size());
	// allocate space for tmp
	if (tmp_contiguous_bboxes_gpu.empty() || tmp_contiguous_bboxes_gpu.size() < num_text_regions)
		tmp_contiguous_bboxes_gpu.reallocate(num_text_regions);
	if (tmp_contiguous_individual_map_gpu.empty() || tmp_contiguous_individual_map_gpu.size() < num_text_regions)
		tmp_contiguous_individual_map_gpu.reallocate(num_text_regions);
	// upload bboxes
	tmp_contiguous_bboxes_gpu.upload(bboxes_contiguous, stream);
	tmp_contiguous_individual_map_gpu.upload(contiguous_individual_map, stream);
	// find max region width
	std::int32_t text_region_width(-1);
	for (auto const& box : bboxes_contiguous)
	{
		auto ratio(static_cast<float>(box.width) / static_cast<float>(box.height));
		auto new_width(static_cast<std::int32_t>(static_cast<float>(text_region_height) * ratio));
		text_region_width = std::max(text_region_width, new_width);
	}
	text_region_width = text_region_height * ((text_region_width - 1) / text_region_height + 1);
	text_region_width = std::min(text_region_width, text_region_max_width);
	assert(text_region_width % 32 == 0);
	// allocate space for output
	if (out_extracted_regions.empty() || out_extracted_regions.size() < num_text_regions * 3 * text_region_height * text_region_width)
		out_extracted_regions.reallocate(num_text_regions * 3 * text_region_height * text_region_width);

	dim3 block(32, 32, 1);
	dim3 grid(text_region_width / 32, text_region_height / 32, num_text_regions);

	//if (num_text_regions == 91)
	//	__debugbreak();

	ExtractTextRegions_kernel<<<grid, block, 0, stream>>>(
		in_frames,
		tmp_contiguous_bboxes_gpu,
		tmp_contiguous_individual_map_gpu,
		out_extracted_regions,
		width,
		height,
		text_region_width,
		text_region_height
		);
	ck2(cudaGetLastError());

	if (out_text_region_width)
		*out_text_region_width = text_region_width;
}
