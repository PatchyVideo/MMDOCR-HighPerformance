#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

void CRAFT(
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> const& in_frames,
	cudawrapper::CUDAHostMemoryUnique<std::uint8_t>& out_text_region_mask_cpu,
	cudawrapper::CUDAHostMemoryUnique<float>& out_text_region_score,
	cudawrapper::CUDADeviceMemoryUnique<float>& tmp_fp32_frames,
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>& tmp_text_region_mask_gpu,
	cudawrapper::CUDADeviceMemoryUnique<float>& tmp_scores,
	nvinfer1::IExecutionContext* craft_context,
	std::int32_t width,
	std::int32_t height,
	std::size_t batch_size,
	std::size_t num_batches,
	std::size_t input_frame_count, // input_frame_count <= batch_size * num_batches
	float text_threshold,
	float link_threshold,
	float low_text,
	cudaEvent_t input_consumed,
	CUstream stream = nullptr
);
