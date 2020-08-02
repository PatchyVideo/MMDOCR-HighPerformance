#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

// (batch) from raw decoded yuv(batch first) to resized, padded and bilateral filtered fp32 RGB image range [-1,1] (NCHW format)
void PreprocessNV12(
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& output,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& output_raw,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> const& batched_yuv_frames,
	std::size_t batch_size,
	std::size_t nv12_frame_size_in_bytes,
	std::int32_t input_width,
	std::int32_t input_height,
	std::int32_t canvas_size,
	std::int32_t * out_output_width,
	std::int32_t * out_output_height,
	CUstream stream = nullptr
);
