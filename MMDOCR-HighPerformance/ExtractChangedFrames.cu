
#include "ExtractChangedFrames.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

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

union load_16bytes
{
	uint4 u128;
	struct
	{
		std::uint8_t u8s[16];
	};
};

// from image of type u8 of shape NCHW (C=3,H and W both are multiple of 32) to image of type u32 of shape NH(W/16)
__global__ void ExtractChangedFrames_stage_1_kernel(
	std::uint8_t const * const in_rgb,
	std::uint32_t *out_intermediate_results,
	std::int32_t frame_height,
	std::int32_t reduced_frame_width
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	load_16bytes const* const in_16bytes(reinterpret_cast<load_16bytes const* const>(in_rgb));
	// step 1: check for boundary
	if (ix >= reduced_frame_width)
		return;
	if (iz == 0)
	{
		write_pixel_gs(out_intermediate_results, 0, ix, iy, 255 * 16 * 3u, reduced_frame_width, frame_height);
	}
	else
	{
		std::uint32_t absdiff(0);
		for (std::int32_t channel(0); channel < 3; ++channel)
		{
			// step 2: load 16 bytes from R channel of image iz-1
			auto img1_segment(read_pixel(in_16bytes, iz - 1, ix, iy, channel, reduced_frame_width, frame_height));
			// step 3: load 16 bytes from R channel of image iz
			auto img2_segment(read_pixel(in_16bytes, iz, ix, iy, channel, reduced_frame_width, frame_height));
			// step 4: compute absdiff
			for (std::int32_t i(0); i < 16; ++i)
				absdiff += std::abs(static_cast<std::int32_t>(img1_segment.u8s[i]) - static_cast<std::int32_t>(img2_segment.u8s[i]));
		}
		// step 5: sum result of 3 channels and store
		write_pixel_gs(out_intermediate_results, iz, ix, iy, absdiff, reduced_frame_width, frame_height);
	}
}

__device__ void reduceBlock(std::uint32_t* sdata, const cg::thread_block& cta)
{
	const unsigned int tid = cta.thread_rank();
	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<std::uint32_t>());
	cg::sync(cta);

	std::uint32_t sum(0);
	if (cta.thread_rank() == 0)
	{
		sum = 0;
		for (int i = 0; i < blockDim.x; i += tile32.size())
		{
			sum += sdata[i];
		}
		sdata[0] = sum;
	}
	cg::sync(cta);
}

// from vector of type u32 of shape N*(H*W/16) to vector of type u32 of shape N
__global__ void ExtractChangedFrames_stage_2_kernel(
	std::uint32_t const * const in_intermediate_results,
	std::uint32_t *out_absdiffs,
	std::int32_t array_length // value is (H*W/16)
)
{
	cg::thread_block block = cg::this_thread_block();
	__shared__ std::uint32_t smem[1024];
	smem[block.thread_rank()] = 0;

	// step 6: reduce image into single thread block
	for (std::int32_t i(block.thread_rank()); i < array_length; i += block.size())
		smem[block.thread_rank()] += in_intermediate_results[i + array_length * blockIdx.x];
	cg::sync(block);
	// step 7: reduce thread block
	reduceBlock(smem, block);
	// step 8: store result
	if (block.thread_rank() == 0)
	{
		out_absdiffs[blockIdx.x] = smem[0];
	}
}

// copy extracted frames to contiguous storage space
__global__ void ExtractChangedFrames_stage_3_kernel(
	std::uint8_t const* const in_frames,
	std::uint8_t* out_frames,
	std::int64_t const* const in_source_index,
	std::int32_t frame_height,
	std::int32_t reduced_frame_width
)
{
	load_16bytes const* const in_16bytes(reinterpret_cast<load_16bytes const* const>(in_frames));
	load_16bytes* out_16bytes(reinterpret_cast<load_16bytes*>(out_frames));
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	if (ix >= reduced_frame_width || iy >= frame_height / 16)
		return;
	std::int64_t source_frame(in_source_index[iz]);
	if (source_frame == -1)
	{
		load_16bytes zeros{};
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, zeros, zeros, zeros, reduced_frame_width, frame_height);
		}
	}
	else
	{
#pragma unroll
		for (int i(0); i < 16; ++i)
		{
			auto r(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 0, reduced_frame_width, frame_height));
			auto g(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 1, reduced_frame_width, frame_height));
			auto b(read_pixel(in_16bytes, source_frame, ix, iy * 16 + i, 2, reduced_frame_width, frame_height));
			write_pixel(out_16bytes, iz, ix, iy * 16 + i, r, g, b, reduced_frame_width, frame_height);
		}
	}
}

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
	CUstream stream
)
{
	std::int32_t reduced_frame_width(frame_width / 16);

	std::size_t intermediate_results_size(input_frame_count * reduced_frame_width * frame_height);
	if (tmp_intermediate_results.empty() || tmp_intermediate_results.size() < intermediate_results_size)
		tmp_intermediate_results.reallocate(intermediate_results_size);

	std::size_t out_absdiffs_size(input_frame_count);
	if (out_absdiffs.empty() || out_absdiffs.size() != input_frame_count)
		out_absdiffs.reallocate(input_frame_count);

	dim3 stage1_block(32, 32, 1);
	dim3 stage1_grid((reduced_frame_width - 1) / 32 + 1, frame_height / 32, input_frame_count);

	// launch stage 1
	ExtractChangedFrames_stage_1_kernel<<<stage1_grid, stage1_block, 0, stream>>>(in_frames, tmp_intermediate_results, frame_height, reduced_frame_width);
	ck2(cudaGetLastError());

	dim3 stage2_block(1024, 1, 1);
	dim3 stage2_grid(input_frame_count, 1, 1);

	// launch stage 2
	ExtractChangedFrames_stage_2_kernel<<<stage2_grid, stage2_block, 0, stream>>>(tmp_intermediate_results, out_absdiffs, frame_height * reduced_frame_width);
	ck2(cudaGetLastError());

	// copy back absdiffs
	out_absdiffs.download_block(out_absdiffs_host, stream);
	// keep frames whose absdiff>threshold
	out_frame_id_map.clear();
	for (std::size_t i(0); i < out_absdiffs_host.size(); ++i)
	{
		if (out_absdiffs_host[i] >= threshold)
		{
			out_frame_id_map.emplace_back(i);
		}
	}
	auto num_frame_kept(out_frame_id_map.size());
	assert(num_frame_kept > 0);
	if (num_frame_kept % output_batch_size != 0)
	{
		auto padded_frames(output_batch_size - (num_frame_kept % output_batch_size));
		for (std::size_t i(0); i < padded_frames; ++i)
			out_frame_id_map.emplace_back(-1);
	}
	out_batches = out_frame_id_map.size() / output_batch_size;

	// copy out_frame_id_map to GPU
	tmp_frame_id_map.upload(out_frame_id_map, stream);

	// allocate space for output frames
	if (out_frames.empty() || out_frames.size() < out_frame_id_map.size() * 3 * frame_width * frame_height)
		out_frames.reallocate(out_frame_id_map.size() * 3 * frame_width * frame_height);

	dim3 stage3_block(32, 32, 1);
	dim3 stage3_grid((reduced_frame_width - 1) / 32 + 1, (frame_height / 16 - 1) / 32 + 1, out_frame_id_map.size());

	// launch stage 3
	ExtractChangedFrames_stage_3_kernel<<<stage3_grid, stage3_block, 0, stream>>>(in_frames, out_frames, tmp_frame_id_map, frame_height, reduced_frame_width);
	ck2(cudaGetLastError());

	// keep first num_frame_kept elements
	out_frame_id_map.resize(num_frame_kept);
	for (auto& e : out_frame_id_map)
		e += first_frame_id; // offset by first_frame_id

	//if (stream)
	//	ck2(cuStreamSynchronize(stream));
}
