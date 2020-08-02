
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <NvInfer.h>

#include "CRAFT.h"
#include "RgbBytesToRgbFp32.h"

template<typename T>
__device__ __forceinline__ void write_pixel_gs(T* grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, T val, std::int32_t width, std::int32_t height)
{
	*(grayscale + width * height * batch + width * y + x) = val;
}

template<typename T>
__device__ __forceinline__ void write_pixel(T* rgb, std::int32_t batch, std::int32_t x, std::int32_t y, T r, T g, T b, std::int32_t width, std::int32_t height)
{
	*(rgb + width * height * 3 * batch + width * height * 0 + width * y + x) = r;
	*(rgb + width * height * 3 * batch + width * height * 1 + width * y + x) = g;
	*(rgb + width * height * 3 * batch + width * height * 2 + width * y + x) = b;
}

template<typename T>
__device__ __forceinline__ T read_pixel(T const* const grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, std::int32_t width, std::int32_t height)
{
	return *(grayscale + width * height * 3 * batch + width * height * channel + width * y + x);
}

template<typename T>
__device__ __forceinline__ T read_pixel_2ch(T const* const grayscale, std::int32_t batch, std::int32_t x, std::int32_t y, std::int32_t channel, std::int32_t width, std::int32_t height)
{
	return *(grayscale + width * height * 2 * batch + width * height * channel + width * y + x);
}

__global__ void CombineCraftScores_kernel(
	float const* const in_scores,
	std::uint8_t* out_mask,
	std::int32_t width,
	std::int32_t height,
	float link_threshold,
	float low_text
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	if (ix >= width || iy >= height)
		return;
	float text_score(read_pixel_2ch(in_scores, iz, ix, iy, 0, width, height));
	float link_score(read_pixel_2ch(in_scores, iz, ix, iy, 1, width, height));
	bool text(text_score > low_text), link(link_score > link_threshold);
	bool comb(text | link);
	write_pixel_gs(out_mask, iz, ix, iy, static_cast<std::uint8_t>(comb * 255), width, height);
}

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
	CUstream stream
)
{
	std::int32_t score_width(width / 2), score_height(height / 2);
	if (tmp_scores.empty() || tmp_scores.size() != batch_size * num_batches * 2 * score_width * score_height)
		tmp_scores.reallocate(batch_size * num_batches * 2 * score_width * score_height);
	for (std::size_t i(0); i < num_batches; ++i)
	{
		// step 1: convert to fp32
		RgbBytesToRgbFp32(reinterpret_cast<std::uint8_t*>(in_frames.at_offset(batch_size * 3 * width * height, i)), tmp_fp32_frames, batch_size, width, height, stream);
		if (stream)
			cuStreamSynchronize(stream);
		// step 2: run CRAFT inference
		std::vector<void*> bindings{ tmp_fp32_frames, reinterpret_cast<float*>(tmp_scores.at_offset(batch_size * 2 * score_width * score_height, i)) };
		
		if (!craft_context->enqueueV2(bindings.data(), stream, std::addressof(input_consumed)))
			throw std::runtime_error("enqueue failed!!!");
		ck2(cudaEventSynchronize(input_consumed));
	}

	// step 3: combine CRAFT result and convert to u8
	if (out_text_region_mask_cpu.empty() || out_text_region_mask_cpu.size() < batch_size * num_batches * 1 * score_width * score_height)
		out_text_region_mask_cpu.reallocate(batch_size * num_batches * 1 * score_width * score_height);
	if (tmp_text_region_mask_gpu.empty() || tmp_text_region_mask_gpu.size() < batch_size * num_batches * 1 * score_width * score_height)
		tmp_text_region_mask_gpu.reallocate(batch_size * num_batches * 1 * score_width * score_height);
	dim3 block(32, 32, 1);
	dim3 grid((score_width - 1) / 32 + 1, (score_height - 1) / 32 + 1, input_frame_count);

	CombineCraftScores_kernel<<<grid, block, 0, stream>>>(tmp_scores, tmp_text_region_mask_gpu, score_width, score_height, link_threshold, low_text);
	ck2(cudaGetLastError());

	tmp_scores.download(out_text_region_score, stream);
	tmp_text_region_mask_gpu.download_block(out_text_region_mask_cpu, stream);
}
