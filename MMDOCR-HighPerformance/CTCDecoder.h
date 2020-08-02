#pragma once

#include "common.h"

#include "CUDAHostMemory.h"

void BuildAlphabet();
std::vector<std::u32string> CTCDecode(cudawrapper::CUDAHostMemoryUnique<std::int32_t> const& ocr_result, std::size_t num_imgs, std::size_t image_width);

