#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

void CompareTextRegions(
	cudawrapper::CUDADeviceMemoryUnique<uchar> const& in_all_text_regions,
	std::vector<comparison_pair> const& in_comparison_pairs,
	cudawrapper::CUDAHostMemoryUnique<std::uint32_t>& out_comparsion_result, // absdiffs
	cudawrapper::CUDADeviceMemoryUnique<comparison_pair>& tmp_comparison_pairs_gpu,
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t>& tmp_comparsion_result_gpu,
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t>& tmp_intermediate_results,
	std::size_t text_region_count,
	std::int32_t region_width,
	std::int32_t region_height,
	CUstream stream = nullptr
);
