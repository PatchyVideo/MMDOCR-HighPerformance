#pragma once

#include <NvInfer.h>

#include "common.h"

#include "CUDADeviceMemory.h"

void OCR(
	cudawrapper::CUDADeviceMemoryUnique<uchar> const& in_text_regions,
	cudawrapper::CUDAHostMemoryUnique<std::int32_t>& out_text_indices,
	cudawrapper::CUDAHostMemoryUnique<float>& out_text_probs,
	cudawrapper::CUDADeviceMemoryUnique<std::int32_t>& tmp_text_indices_gpu,
	cudawrapper::CUDADeviceMemoryUnique<float>& tmp_text_probs_gpu,
	cudawrapper::CUDADeviceMemoryUnique<float>& tmp_fp32_frames,
	nvinfer1::IExecutionContext* ocr_context,
	std::int32_t width,
	std::int32_t height,
	std::size_t batch_size,
	std::size_t num_batches,
	std::size_t k,
	std::size_t input_region_count, // input_frame_count <= batch_size * num_batches
	cudaEvent_t input_consumed,
	CUstream stream = nullptr
);
