#pragma once

#include "common.h"

#include "CUDAHostMemory.h"

void BuildAlphabet();
void BuildBigramProbs();
std::vector<std::u32string> CTCDecode(
	cudawrapper::CUDAHostMemoryUnique<std::int32_t> const& ocr_result_indices,
	cudawrapper::CUDAHostMemoryUnique<float> const& ocr_result_probs,
	std::size_t num_imgs,
	std::size_t image_width,
	std::size_t k
);

