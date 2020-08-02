
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "CompareTextRegions.h"

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

__global__ void CompareTextRegions_stage_1_kernel(
	uchar const* const regions,
	comparison_pair const* const compares,
	std::uint32_t* out_intermediate_results,
	std::int32_t reduced_region_width,
	std::int32_t region_height
)
{
	std::int32_t ix(blockIdx.x * blockDim.x + threadIdx.x), iy(blockIdx.y * blockDim.y + threadIdx.y), iz(blockIdx.z);
	auto img1_idx(compares[iz].first), img2_idx(compares[iz].second);
	load_16bytes const* const in_16bytes(reinterpret_cast<load_16bytes const* const>(regions));
	if (ix >= reduced_region_width)
		return;
	std::uint32_t absdiff(0);
	for (std::int32_t channel(0); channel < 3; ++channel)
	{
		// step 2: load 16 bytes from R channel of image img1
		auto img1_segment(read_pixel(in_16bytes, img1_idx, ix, iy, channel, reduced_region_width, region_height));
		// step 3: load 16 bytes from R channel of image img2
		auto img2_segment(read_pixel(in_16bytes, img2_idx, ix, iy, channel, reduced_region_width, region_height));
		// step 4: compute absdiff
		for (std::int32_t i(0); i < 16; ++i)
			absdiff += std::abs(static_cast<std::int32_t>(img1_segment.u8s[i]) - static_cast<std::int32_t>(img2_segment.u8s[i]));
	}
	// step 5: sum result of 3 channels and store
	write_pixel_gs(out_intermediate_results, iz, ix, iy, absdiff, reduced_region_width, region_height);
}

__device__ void reduceBlock_2(std::uint32_t* sdata, const cg::thread_block& cta)
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

__global__ void CompareTextRegions_stage_2_kernel(
	std::uint32_t const* const in_intermediate_results,
	std::uint32_t* out_absdiffs,
	std::int32_t array_length // value is (H*W/16)
)
{
	cg::thread_block block = cg::this_thread_block();
	__shared__ std::uint32_t smem[1024];
	smem[block.thread_rank()] = 0;

	for (std::int32_t i(block.thread_rank()); i < array_length; i += block.size())
		smem[block.thread_rank()] += in_intermediate_results[i + array_length * blockIdx.x];
	cg::sync(block);
	reduceBlock_2(smem, block);
	if (block.thread_rank() == 0)
	{
		out_absdiffs[blockIdx.x] = smem[0];
	}
}

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
	CUstream stream
)
{
	std::int32_t reduced_region_width(region_width / 16);
	std::size_t num_comparisons(in_comparison_pairs.size());

	std::size_t intermediate_results_size(num_comparisons * region_width * region_height);
	if (tmp_intermediate_results.empty() || tmp_intermediate_results.size() < intermediate_results_size)
		tmp_intermediate_results.reallocate(intermediate_results_size);

	std::size_t out_absdiffs_size(num_comparisons);
	if (out_comparsion_result.empty() || out_comparsion_result.size() != num_comparisons)
		out_comparsion_result.reallocate(num_comparisons);
	if (tmp_comparsion_result_gpu.empty() || tmp_comparsion_result_gpu.size() != num_comparisons)
		tmp_comparsion_result_gpu.reallocate(num_comparisons);

	tmp_comparison_pairs_gpu.upload(in_comparison_pairs, stream);

	dim3 stage1_block(32, 32, 1);
	dim3 stage1_grid((reduced_region_width - 1) / 32 + 1, region_height / 32, num_comparisons);

	// launch stage 1
	CompareTextRegions_stage_1_kernel<<<stage1_grid, stage1_block, 0, stream>>>
		(in_all_text_regions, tmp_comparison_pairs_gpu, tmp_intermediate_results, reduced_region_width, region_height);
	ck2(cudaGetLastError());

	dim3 stage2_block(1024, 1, 1);
	dim3 stage2_grid(num_comparisons, 1, 1);

	// launch stage 2
	CompareTextRegions_stage_2_kernel<<<stage2_grid, stage2_block, 0, stream>>>(tmp_intermediate_results, tmp_comparsion_result_gpu, reduced_region_width * region_height);
	ck2(cudaGetLastError());

	// wait for result
	if (stream)
		ck2(cuStreamSynchronize(stream));

	// copy back absdiffs
	tmp_comparsion_result_gpu.download_block(out_comparsion_result, stream);
}
