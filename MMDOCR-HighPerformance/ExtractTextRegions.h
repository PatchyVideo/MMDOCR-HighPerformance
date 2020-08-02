#pragma once

#include "common.h"

#include "CUDADeviceMemory.h"

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
	std::int32_t *out_text_region_width = nullptr,
	CUstream stream = nullptr
);
