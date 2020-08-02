#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

void RgbBytesToRgbFp32(
	std::uint8_t const* const input_u8_frames,
	cudawrapper::CUDADeviceMemoryUnique<float>& out_fp32_frames,
	std::size_t input_frame_count,
	std::int32_t width,
	std::int32_t height,
	CUstream stream = nullptr
);
