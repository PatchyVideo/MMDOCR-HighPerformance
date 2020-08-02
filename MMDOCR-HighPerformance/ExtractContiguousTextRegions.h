#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

void ExtractContiguousTextRegions(
	cudawrapper::CUDADeviceMemoryUnique<uchar> const& in_all_regions,
	std::vector<std::int32_t> in_region_indices,
	cudawrapper::CUDADeviceMemoryUnique<uchar>& out_contiguous_regions,
	std::int32_t& out_batches,
	cudawrapper::CUDADeviceMemoryUnique<std::int32_t>& tmp_region_indices_gpu,
	std::int32_t region_width,
	std::int32_t region_height,
	std::size_t output_batch_size,
	CUstream stream = nullptr
);
