#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

void ExtractChangedFrames(
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> const& in_frames,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& out_frames,
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t>& tmp_intermediate_results,
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t>& out_absdiffs,
	cudawrapper::CUDADeviceMemoryUnique<std::int64_t>& tmp_frame_id_map,
	cudawrapper::CUDAHostMemoryUnique<std::uint32_t>& out_absdiffs_host,
	std::vector<std::int64_t>& out_frame_id_map,
	std::int32_t& out_batches,
	std::int64_t first_frame_id,
	std::size_t input_frame_count,
	std::int32_t frame_width,
	std::int32_t frame_height,
	std::size_t output_batch_size,
	std::uint32_t threshold,
	CUstream stream = nullptr
);
