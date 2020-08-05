
#include "OCR.h"
#include "RgbBytesToRgbFp32.h"

void OCR(
	cudawrapper::CUDADeviceMemoryUnique<uchar> const& in_text_regions,
	cudawrapper::CUDAHostMemoryUnique<std::int32_t>& out_text_indices,
	cudawrapper::CUDADeviceMemoryUnique<std::int32_t>& tmp_text_indices_gpu,
	cudawrapper::CUDADeviceMemoryUnique<float>& tmp_fp32_frames,
	nvinfer1::IExecutionContext* ocr_context,
	std::int32_t width,
	std::int32_t height,
	std::size_t batch_size,
	std::size_t num_batches,
	std::size_t input_region_count, // input_frame_count <= batch_size * num_batches
	cudaEvent_t input_consumed,
	CUstream stream
)
{
	std::int32_t num_text_idx_per_region(width / 4 + 1);
	if (tmp_text_indices_gpu.empty() || tmp_text_indices_gpu.size() < num_batches * batch_size * num_text_idx_per_region)
		tmp_text_indices_gpu.reallocate(num_batches * batch_size * num_text_idx_per_region);
	for (std::size_t i(0); i < num_batches; ++i)
	{
		// step 1: convert to fp32
		RgbBytesToRgbFp32(reinterpret_cast<std::uint8_t*>(in_text_regions.at_offset(batch_size * 3 * width * height, i)), tmp_fp32_frames, batch_size, width, height, stream);
		//if (stream)
		//	cuStreamSynchronize(stream);
		// step 2: run OCR inference
		std::vector<void*> bindings{ tmp_fp32_frames, reinterpret_cast<std::int32_t*>(tmp_text_indices_gpu.at_offset(batch_size * num_text_idx_per_region, i)) };

		if (!ocr_context->enqueueV2(bindings.data(), stream, std::addressof(input_consumed)))
			throw std::runtime_error("enqueue failed!!!");
		ck2(cudaEventSynchronize(input_consumed));
	}

	tmp_text_indices_gpu.download_block(out_text_indices, stream);
}
