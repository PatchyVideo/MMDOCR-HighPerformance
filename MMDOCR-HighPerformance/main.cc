
#include <iostream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <condition_variable>
#include "NvCodecs/NvDecoder/NvDecoder.h"
#include "NvCodecs/Utils/NvCodecUtils.h"
#include "NvCodecs/Utils/FFmpegDemuxer.h"
#include "NvCodecs/Utils/ColorSpace.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "lodepng.h"

#include "CUDAContext.h"
#include "CUDAStream.h"
#include "CUDAThreadContext.h"
#include "CUDAEvent.h"
#include "NvInferRuntime.h"
#include "NvInferEngine.h"
#include "NvInferContext.h"

#include "PreprocessNV12.h"
#include "ExtractChangedFrames.h"
#include "CRAFT.h"
#include "ExtractTextRegions.h"
#include "CompareTextRegions.h"
#include "ExtractContiguousTextRegions.h"
#include "OCR.h"
#include "CTCDecoder.h"
#include "SubtitleGenerator.h"

#include "FPSCounter.h"
#include "BBoxUtils.h"

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();


class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		//if (severity != Severity::kINFO)
		std::cout << msg << std::endl;
	}
} g_trtLogger;

void test_ExtractChangedFrames()
{
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> in_frames(32 * 3 * 640 * 384);
	cudaMemset((void*)(in_frames), 0, 32 * 3 * 640 * 384);
	cudaMemset((void*)(in_frames + 10 * 3 * 640 * 384), 255, 3 * 640 * 384);
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> out_frames;
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t> tmp1;
	cudawrapper::CUDADeviceMemoryUnique<std::int64_t> tmp2;
	cudawrapper::CUDADeviceMemoryUnique<std::uint32_t> absdiffs_gpu;
	cudawrapper::CUDAHostMemoryUnique<std::uint32_t> absdiffs_cpu;
	std::vector<int64_t> frame_id_mapping;
	std::int32_t batches;
	ExtractChangedFrames(
		in_frames,
		out_frames,
		tmp1,
		absdiffs_gpu,
		tmp2,
		absdiffs_cpu,
		frame_id_mapping,
		batches,
		0,
		32,
		640,
		384,
		8,
		114514
	);
	__debugbreak();
}

struct FrameBuffer;
struct WorkerThread;

struct FrameBuffer
{
	cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> data;
	enum class FrameBufferState
	{
		Uninitialized,
		Ready,
		Using
	} state;
	std::size_t count;
	std::size_t capacity;
	std::size_t element_size;
	WorkerThread* associated_worker;
	cudaVideoSurfaceFormat format;
	std::size_t start_frame;
	std::int32_t input_width, input_height;
	FrameBuffer() :
		state(FrameBufferState::Uninitialized),
		count(0),
		capacity(0),
		element_size(0),
		associated_worker(nullptr),
		format(cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12),
		start_frame(0),
		input_width(0),
		input_height(0)
	{

	}
	void Init(std::size_t num_frame, std::size_t frame_size, cudaVideoSurfaceFormat vid_format, std::int32_t width, std::int32_t height)
	{
		if (state == FrameBufferState::Uninitialized)
		{
			data = std::move(cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>(frame_size * num_frame));
			capacity = num_frame;
			element_size = frame_size;
			format = vid_format;
			input_width = width;
			input_height = height;
			state = FrameBufferState::Ready;
		}
	}
	void Clear()
	{
		count = 0;
	}
	// returns how many frames not uploaded
	std::tuple<std::size_t, bool> Copy(std::uint8_t* frames[], std::size_t num_frames)
	{
		std::size_t remaining_spaces(capacity - count);
		std::size_t frames_to_copy(std::min(num_frames, remaining_spaces));
		for (std::size_t i(0); i < frames_to_copy; ++i)
			ck2(cuMemcpy(data.at_offset(element_size, count + i), (CUdeviceptr)frames[i], element_size));
		count += frames_to_copy;
		return { num_frames - frames_to_copy, count >= capacity };
	}
	FrameBuffer(FrameBuffer const& a) = delete;
	FrameBuffer& operator=(FrameBuffer const& a) = delete;
	FrameBuffer(FrameBuffer&& a) = delete;
	FrameBuffer& operator=(FrameBuffer&& a) = delete;
};

std::mutex g_mutex, g_subtitleMutex;

void chw2hwc(cv::Mat& out, cudawrapper::CUDAHostMemoryUnique<std::uint8_t> const& src, std::int32_t width, std::int32_t height, std::size_t batch)
{
	out.create(cv::Size(width, height), CV_8UC3);
	auto pixel_buffer(reinterpret_cast<uchar3*>(out.data));
	for (std::int32_t x(0); x < width; ++x)
		for (std::int32_t y(0); y < height; ++y)
		{
			auto r(src[width * height * 3 * batch + width * height * 0 + width * y + x]);
			auto g(src[width * height * 3 * batch + width * height * 1 + width * y + x]);
			auto b(src[width * height * 3 * batch + width * height * 2 + width * y + x]);
			pixel_buffer[width * y + x].x = b;
			pixel_buffer[width * y + x].y = g;
			pixel_buffer[width * y + x].z = r;
		}
}

nvinfer1::ICudaEngine* LoadEngineFromFile(std::string_view filename, nvinfer1::IRuntime* runtime)
{
	std::lock_guard<std::mutex> guard(g_mutex);
	std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (file.read(buffer.data(), size))
	{
		nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
		return engine;
	}
	return nullptr;
}

struct WorkerThread
{
	FrameBuffer* associated_frame_buffer;
	bool exit;
	std::size_t rank;
	double video_fps;

	std::thread thread;
	CUcontext context;
	nvinfer1::IRuntime* trt_runtime;
	SubtitleGenerator *generator;

	enum class ThreadState
	{
		Uninitialized = 0,
		Idle = 1,
		Waiting = 2,
		Working = 3,
		Exited = 4,
		Error = 5
	} state;

	float text_threshold,
		link_threshold,
		low_text;
	float cost_scale;

	WorkerThread() : associated_frame_buffer(nullptr), exit(false), rank(0), state(ThreadState::Uninitialized)
	{
		text_threshold = 0.99f;
		link_threshold = 0.1f;
		low_text = 0.5f;
		cost_scale = 1000.0f;
	}

	void Spawn(std::size_t r, CUcontext ctx, nvinfer1::IRuntime* trt_runtime, double fps, SubtitleGenerator *gen)
	{
		rank = r;
		context = ctx;
		video_fps = fps;
		this->trt_runtime = trt_runtime;
		generator = gen;
		thread = std::thread(&WorkerThread::mainloop, this);
	}

	void SetState(ThreadState val)
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		state = val;
	}

	ThreadState Poll()
	{
		return state;
	}
	void ConsumeInput()
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		associated_frame_buffer->associated_worker = nullptr;
		associated_frame_buffer->Clear();
		associated_frame_buffer = nullptr;
	}

	void MasterWaitForIdle()
	{
		for (;;)
		{
			bool idle(false);
			{
				std::lock_guard<std::mutex> guard(g_mutex);
				if (state == ThreadState::Idle)
					idle = true;
			}
			if (idle)
			{
				return;
			} else
			{
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(2ms);
				//std::this_thread::yield();
			}
		}
	}

	void WaitForMessage()
	{
		for (;;)
		{
			bool received(false);
			{
				std::lock_guard<std::mutex> guard(g_mutex);
				if (associated_frame_buffer || exit)
					received = true;
			}
			if (received)
			{
				std::lock_guard<std::mutex> guard(g_mutex);
				state = ThreadState::Waiting;
				break;
			}
			else
			{
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(2ms);
				//std::this_thread::yield();
			}
		}
	}

	void Join()
	{
		thread.join();
	}

	void NotifyExit()
	{
		exit = true;
	}

	void SendWork(FrameBuffer& buffer)
	{
		std::lock_guard<std::mutex> guard(g_mutex);
		assert(((state == ThreadState::Idle) ^ (associated_frame_buffer == nullptr)) == 0);
		assert(state == ThreadState::Idle);
		assert(associated_frame_buffer == nullptr);
		associated_frame_buffer = std::addressof(buffer);
		associated_frame_buffer->associated_worker = this;
		state = ThreadState::Waiting;
	}

	void InsertSubtitle(std::uint32_t frame_id, std::vector<std::u32string> const& texts)
	{
		std::size_t frame_ms(static_cast<std::size_t>(static_cast<double>(frame_id) / video_fps * 1000.0));
		std::lock_guard<std::mutex> guard(g_subtitleMutex);
		generator->InsertSubtitle(frame_ms, texts);
	}

	WorkerThread(WorkerThread const& a) = delete;
	WorkerThread& operator=(WorkerThread const& a) = delete;
	WorkerThread(WorkerThread&& a) = delete;
	WorkerThread& operator=(WorkerThread&& a) = delete;

	void mainloop()
	{
		cudawrapper::NvInferEngine craft_engine(LoadEngineFromFile("detect.trt", trt_runtime));
		cudawrapper::NvInferEngine ocr_engine(LoadEngineFromFile("ocr.trt", trt_runtime));
		cudawrapper::CUDAThreadContext context_scope(context);
		cudawrapper::CUDAStream stream;
		cudawrapper::CUDAEvent input_consumed;

		cudawrapper::CUDAHostMemoryUnique<std::uint8_t> visualize_tmp;
		cudawrapper::CUDADeviceMemoryUnique<std::uint32_t> tmp_reduction_intermediate_results;
		cudawrapper::CUDADeviceMemoryUnique<float> tmp_fp32_images;

		cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> preprocess_output, preprocess_raw_output;
		std::int32_t width, height;

		cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> detect_frame_change_output;
		cudawrapper::CUDADeviceMemoryUnique<std::int64_t> detect_frame_change_tmp2;
		cudawrapper::CUDADeviceMemoryUnique<std::uint32_t> detect_frame_change_absdiffs_gpu;
		cudawrapper::CUDAHostMemoryUnique<std::uint32_t> detect_frame_change_absdiffs_cpu;
		std::vector<int64_t> detect_frame_change_frame_id_mapping;
		std::int32_t detect_frame_change_batches;

		std::size_t const detector_batch_size(8);
		cudawrapper::NvInferContext craft_context(craft_engine.createContext());
		
		std::size_t const ocr_batch_size(16);
		cudawrapper::NvInferContext ocr_context(ocr_engine.createContext());

		cudawrapper::CUDADeviceMemoryUnique<float> craft_fp32_scores;
		cudawrapper::CUDAHostMemoryUnique<float> craft_fp32_scores_cpu;
		cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> craft_mask_gpu;
		cudawrapper::CUDAHostMemoryUnique<std::uint8_t> craft_mask_cpu;

		cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> text_regions;
		cudawrapper::CUDADeviceMemoryUnique<BBox> text_region_tmp_bboxes_gpu;
		cudawrapper::CUDADeviceMemoryUnique<subtitle_index> text_region_tmp_subtitle_index_gpu;

		cudawrapper::CUDADeviceMemoryUnique<comparison_pair> compare_regions_comparison_pairs_gpu;
		cudawrapper::CUDADeviceMemoryUnique<std::uint32_t> compare_regions_comparison_result_gpu;
		cudawrapper::CUDAHostMemoryUnique<std::uint32_t> compare_regions_comparison_result;

		cudawrapper::CUDADeviceMemoryUnique<uchar> contiguous_text_regions;
		cudawrapper::CUDADeviceMemoryUnique<std::int32_t> contiguous_text_regions_index_tmp;
		std::int32_t contiguous_text_regions_batches;

		cudawrapper::CUDADeviceMemoryUnique<std::int32_t> ocr_text_indices_gpu;
		cudawrapper::CUDAHostMemoryUnique<std::int32_t> ocr_text_indices;

		for (;;)
		{
			SetState(ThreadState::Idle);
			WaitForMessage();
			if (associated_frame_buffer == nullptr && exit)
			{
				// we are done
				SetState(WorkerThread::ThreadState::Exited);
				break;
			}
			SetState(ThreadState::Working);
			// work here
			std::size_t start_frame(associated_frame_buffer->start_frame);
			std::int32_t input_width(associated_frame_buffer->input_width), input_height(associated_frame_buffer->input_height);
			std::size_t input_frame_count(associated_frame_buffer->count);
			std::size_t input_frame_size(associated_frame_buffer->element_size);

			// step 1: process input
			PreprocessNV12(
				preprocess_output,
				preprocess_raw_output,
				associated_frame_buffer->data,
				associated_frame_buffer->count,
				associated_frame_buffer->element_size,
				associated_frame_buffer->input_width,
				associated_frame_buffer->input_height,
				640,
				std::addressof(width),
				std::addressof(height),
				stream
			);
			ck2(cuStreamSynchronize(stream));
			ConsumeInput();

			// step 2: find changed frames
			ExtractChangedFrames(
				preprocess_output,
				detect_frame_change_output,
				tmp_reduction_intermediate_results,
				detect_frame_change_absdiffs_gpu,
				detect_frame_change_tmp2,
				detect_frame_change_absdiffs_cpu,
				detect_frame_change_frame_id_mapping,
				detect_frame_change_batches,
				start_frame,
				input_frame_count,
				width,
				height,
				detector_batch_size,	// batch size for subsequence inference
				width * height,			// threshold
				stream
			);

			std::size_t extract_frame_count(detect_frame_change_frame_id_mapping.size());

			// step 3: run CRAFT text detector
			craft_context.context->setOptimizationProfile(0);
			craft_context.context->setBindingDimensions(0, nvinfer1::Dims4(detector_batch_size, 3, height, width));
			CRAFT(
				detect_frame_change_output,
				craft_mask_cpu,
				craft_fp32_scores_cpu,
				tmp_fp32_images,
				craft_mask_gpu,
				craft_fp32_scores,
				craft_context,
				width,
				height,
				detector_batch_size,
				detect_frame_change_batches,
				extract_frame_count,
				text_threshold,
				link_threshold,
				low_text,
				input_consumed,
				stream
			);

			// step 4: extract text bboxes
			//detect_frame_change_output.download_block(visualize_tmp, stream);
			std::vector<std::vector<BBox>> bboxes(extract_frame_count);
			for (std::size_t i(0); i < extract_frame_count; ++i)
			{
				cv::Mat score_comb(height / 2, width / 2, CV_8UC1, craft_mask_cpu.at_offset(height / 2 * width / 2, i));
				cv::Mat score_fp32(height / 2, width / 2, CV_32FC1, craft_fp32_scores_cpu.at_offset(2 * height / 2 * width / 2, i));
				thread_local static cv::Mat labels;
				cv::Mat stats;
				cv::Mat centroids;
				int num_labels(cv::connectedComponentsWithStats(score_comb, labels, stats, centroids, 4));
				bboxes[i].reserve(num_labels);
				for (int j(0); j < num_labels; ++j)
				{
					auto area(stats.at<int>(cv::Point(4, j)));
					if (area < 10)
						continue;

					cv::Mat label_mask;
					cv::compare(labels, j, label_mask, cv::CmpTypes::CMP_EQ);
					label_mask.convertTo(label_mask, CV_32F);
					cv::Mat masked_score;
					cv::multiply(score_fp32, label_mask, masked_score);
					double minV, maxV;
					cv::minMaxLoc(masked_score, std::addressof(minV), std::addressof(maxV));
					if (static_cast<float>(maxV) < text_threshold)
						continue;

					auto x(stats.at<int>(cv::Point(0, j)));
					auto y(stats.at<int>(cv::Point(1, j)));
					auto w(stats.at<int>(cv::Point(2, j)));
					auto h(stats.at<int>(cv::Point(3, j)));

					if (x >= score_comb.cols || y >= score_comb.rows || x < 0 || y < 0 || w <= 0 || h <= 0 || (x == 0 && y == 0 && w == score_comb.cols && h == score_comb.rows))
						continue;

					auto niter(static_cast<std::int32_t>(std::sqrt(area * std::min(w, h) / (w * h)) * 2));
					auto extend(niter + static_cast<std::int32_t>(static_cast<float>(std::min(w, h)) * 0.07f));

					bboxes[i].emplace_back(
						static_cast<std::int32_t>(x - extend) * 2,
						static_cast<std::int32_t>(y - extend) * 2,
						static_cast<std::int32_t>(w + extend * 2) * 2,
						static_cast<std::int32_t>(h + extend * 2) * 2
					);
				}

				bboxes[i] = MergeBBoxes(MergeBBoxes(bboxes[i]));
				for (auto& box : bboxes[i])
					box.crop(width, height);

				//cv::Mat frame;
				//chw2hwc(frame, visualize_tmp, width, height, i);
				//for (auto const& box : bboxes[i])
				//{
				//	cv:rectangle(frame, box.rect(), cv::Scalar(255, 0, 0), 2);
				//}
				//cv::imshow("scores", score_comb);
				//cv::imshow("text regions", frame);
				//cv::waitKey(1);
			}

			// step 5: extract text region from bboxes
			std::vector<subtitle_index> contiguous_individual_map;
			std::vector<BBox> bboxes_contiguous;
			std::unordered_map<subtitle_index, std::uint32_t, pair_hash> individual_contiguous_map;
			std::uint32_t subtitle_index_counter(0);
			for (auto const& [i, frame_boxes] : enumerate(bboxes))
			{
				for (auto const& [j, box] : enumerate(frame_boxes))
				{
					bboxes_contiguous.emplace_back(box);
					contiguous_individual_map.emplace_back(i, j);
					individual_contiguous_map[{i, j}] = subtitle_index_counter++;
				}
			}
			std::size_t const num_text_regions(subtitle_index_counter);
			std::int32_t text_region_width(0);

			if (num_text_regions > 0)
			{
				ExtractTextRegions(
					detect_frame_change_output,
					text_regions,
					text_region_tmp_bboxes_gpu,
					text_region_tmp_subtitle_index_gpu,
					bboxes_contiguous,
					contiguous_individual_map,
					width,
					height,
					32,
					640,
					std::addressof(text_region_width),
					stream
				);

				// step 6: extract changed text regions
				// step 6.1: generate a set of candidates to compare
				// step 6.1.1: generate assignments
				std::vector<std::vector<int>> assignments(extract_frame_count - 1);
				std::vector<comparison_pair> text_region_comparison_candidates;
				std::vector<std::int32_t> text_region_to_ocr;
				std::unordered_map<std::int32_t, std::size_t> text_region_ocr_index_map;
				// text regions in frame 0 need OCR
				for (std::size_t i(0); i < bboxes[0].size(); ++i)
				{
					text_region_to_ocr.emplace_back(individual_contiguous_map[{0, i}]);
					text_region_ocr_index_map[individual_contiguous_map[{0, i}]] = text_region_to_ocr.size() - 1;
				}
				for (std::size_t cur_frame(1); cur_frame < extract_frame_count; ++cur_frame)
				{
					auto const& boxA(bboxes[cur_frame - 1]);
					auto const& boxB(bboxes[cur_frame]);
					auto M(boxA.size()), N(boxB.size());
					std::vector<int> row_assignments;
					if (M > 0 && N > 0)
					{
						auto costmat(std::make_unique<int[]>(M * N));
						for (std::size_t i(0); i < M; ++i)
							for (std::size_t j(0); j < N; ++j)
							{
								costmat[i * N + j] = static_cast<int>((1.0f - boxA[i].IoU(boxB[j])) * cost_scale);
							}
						row_assignments = FindMinCostAssignment(costmat.get(), M, N);
					}
					assignments[cur_frame - 1] = row_assignments;
					std::vector<std::int32_t> boxB_assigned(N, 0);
					if(row_assignments.size())
						for (std::size_t i(0); i < M; ++i)
						{
							auto j(row_assignments[i]);
							if (j != -1)
							{
								auto u(individual_contiguous_map[{cur_frame - 1, i}]);
								auto v(individual_contiguous_map[{cur_frame, j}]);
								text_region_comparison_candidates.emplace_back(u, v);
								boxB_assigned[j] = 1;
							}
						}
					for (std::size_t j(0); j < N; ++j)
					{
						if (!boxB_assigned[j])
						{
							// not assigned, means a new trajectory
							text_region_to_ocr.emplace_back(individual_contiguous_map[{cur_frame, j}]);
							text_region_ocr_index_map[individual_contiguous_map[{cur_frame, j}]] = text_region_to_ocr.size() - 1;
						}
					}
				}
				if (text_region_comparison_candidates.size())
				{
					// step 6.1.2: make comparsion
					CompareTextRegions(
						text_regions,
						text_region_comparison_candidates,
						compare_regions_comparison_result,
						compare_regions_comparison_pairs_gpu,
						compare_regions_comparison_result_gpu,
						tmp_reduction_intermediate_results,
						num_text_regions,
						text_region_width,
						32,
						stream
					);
					// step 6.2: run inter-frame text region change detection algo, using cached comparsion result
					for (std::size_t i(0); i < text_region_comparison_candidates.size(); ++i)
					{
						double absdiff(static_cast<double>(compare_regions_comparison_result[i]));
						absdiff /= static_cast<double>(text_region_width * 32 * 3);
						if (absdiff >= 0.7)
						{
							// text region changed, mark v as beginning of new trajectory
							// meaning it needs to go through OCR
							auto [u, v] = text_region_comparison_candidates[i];
							text_region_to_ocr.emplace_back(v);
							text_region_ocr_index_map[v] = text_region_to_ocr.size() - 1;
							auto [from_frame_idx, from_region_idx] = contiguous_individual_map[u];
							auto [to_frame_idx, to_region_idx] = contiguous_individual_map[v];
							assignments[from_frame_idx][from_region_idx] = -1; // remove edge
						}
					}
				}
				if (text_region_to_ocr.size())
				{
					// step 6.3: create contiguous text regions
					ExtractContiguousTextRegions(
						text_regions,
						text_region_to_ocr,
						contiguous_text_regions,
						contiguous_text_regions_batches,
						contiguous_text_regions_index_tmp,
						text_region_width,
						32,
						ocr_batch_size,
						stream
					);
					
					// step 7: run OCR
					ocr_context.context->setOptimizationProfile(0);
					ocr_context.context->setBindingDimensions(0, nvinfer1::Dims4(ocr_batch_size, 3, 32, text_region_width));
					OCR(
						contiguous_text_regions,
						ocr_text_indices,
						ocr_text_indices_gpu,
						tmp_fp32_images,
						ocr_context,
						text_region_width,
						32,
						ocr_batch_size,
						contiguous_text_regions_batches,
						text_region_to_ocr.size(),
						input_consumed,
						stream
					);

					//ocr_text_indices.reallocate(ocr_batch_size * contiguous_text_regions_batches * (text_region_width / 4 + 1));
					//memset(ocr_text_indices, 0, ocr_text_indices.size_bytes());


					auto ocr_result(CTCDecode(ocr_text_indices, text_region_to_ocr.size(), text_region_width / 4 + 1));
					//for (auto const& s : ocr_result) {
					//	std::cout << ConvertU32toU8(s) << "\n";
					//}
					//contiguous_text_regions.download_block(visualize_tmp, stream);
					//for (std::size_t i(0); i < text_region_to_ocr.size(); ++i)
					//{
					//	cv::Mat region;
					//	chw2hwc(region, visualize_tmp, text_region_width, 32, i);
					//	cv::imshow("text region", region);
					//	cv::waitKey(10);
					//}
					// handle first frame
					std::vector<std::u32string> last_frame_texts;
					for (std::size_t i(0); i < bboxes[0].size(); ++i)
					{
						auto contiguous_index(individual_contiguous_map[{0, i}]);
						assert(text_region_ocr_index_map.count(contiguous_index));
						last_frame_texts.emplace_back(ocr_result[text_region_ocr_index_map[contiguous_index]]);
					}
					// step 8: send to global subtitle buffer
					InsertSubtitle(detect_frame_change_frame_id_mapping[0], last_frame_texts);
					// handle remaining frames
					for (std::size_t cur_frame(1); cur_frame < extract_frame_count; ++cur_frame)
					{
						//if (detect_frame_change_frame_id_mapping[cur_frame] == 92)
						//	__debugbreak();
						auto const& boxA(bboxes[cur_frame - 1]);
						auto const& boxB(bboxes[cur_frame]);
						auto M(boxA.size()), N(boxB.size());
						std::vector<int> row_assignments(assignments[cur_frame - 1]);
						std::vector<int> col_assignments(N, -1);
						if (row_assignments.size())
							for (int i(0); i < M; ++i)
								if (row_assignments[i] != -1)
									col_assignments[row_assignments[i]] = i;
						std::vector<std::u32string> cur_frame_texts;
						bool texts_changed_since_last_frame(M != N);
						for (int i(0); i < N; ++i) // for each text of current frame
						{
							// try find a text in ocr_result
							auto contiguous_index(individual_contiguous_map[{cur_frame, i}]);
							if (text_region_ocr_index_map.count(contiguous_index))
							{
								auto ocr_index(text_region_ocr_index_map[contiguous_index]);
								//cv::Mat region;
								//chw2hwc(region, visualize_tmp, text_region_width, 32, ocr_index);
								//cv::imshow("text region", region);
								//cv::waitKey(1);
								cur_frame_texts.emplace_back(ocr_result[ocr_index]);
								texts_changed_since_last_frame = true;
							}
							else
							{
								// if not, find from previous texts
								if (col_assignments[i] != -1 && col_assignments[i] < last_frame_texts.size())
								{
									cur_frame_texts.emplace_back(last_frame_texts[col_assignments[i]]);
								}
								else
								{
									// bug
									assert(false);
								}
							}
						}
						if (texts_changed_since_last_frame)
						{
							// step 8: send to global subtitle buffer
							InsertSubtitle(detect_frame_change_frame_id_mapping[cur_frame], cur_frame_texts);
						}
						last_frame_texts = cur_frame_texts;
					}

				}
			}
		}
	}
};

std::size_t const NUM_THREADS = 1;
std::size_t const DECODE_BUFFER_SIZE = 1;
std::size_t const DECODE_BATCH_SIZE = 256;

int main()
{
	ck2(cuInit(0));
	cudawrapper::CUDAContext context(0, 0);

	BuildAlphabet();

	cudawrapper::NvInferRuntime trt_runtime(g_trtLogger);
	//auto craft_engine(LoadEngineFromFile("detect.trt", trt_runtime.runtime));
	//craft_engine->

	//test_ExtractChangedFrames();
	//return 0;

	SubtitleGenerator sub_gen;
	FFmpegDemuxer demuxer("sanae.mp4");
	std::cout << "Video width: " << demuxer.GetWidth() << " height: " << demuxer.GetHeight() << " duration(us): " << demuxer.GetDuration() << " frame count: " << demuxer.GetFrameCount() << " FPS: " << demuxer.GetFPS() << "\n";

	NvDecoder dec(context, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
	int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
	uint8_t* pVideo = NULL, ** ppFrame;
	bool bDecodeOutSemiPlanar = false;

	std::vector<FrameBuffer> frame_buffer(DECODE_BUFFER_SIZE);
	FrameBuffer* cur_frame_buffer(std::addressof(frame_buffer.front()));
	bool inited(false);

	std::vector<WorkerThread> workers(NUM_THREADS);
	for (std::size_t i(0); i < NUM_THREADS; ++i)
		workers[i].Spawn(i, context, trt_runtime.runtime, demuxer.GetFPS(), std::addressof(sub_gen));

	auto FindIdleWorker([&workers]() -> WorkerThread*{
		for (;;)
		{
			bool found(false);
			WorkerThread* t(nullptr);
			{
				std::lock_guard<std::mutex> guard(g_mutex);
				for (auto& w : workers)
				{
					if (w.Poll() == WorkerThread::ThreadState::Idle)
					{
						found = true;
						t = std::addressof(w);
					}
				}
			}
			if (found)
				return t;
			else
			{
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(2ms);
				//std::this_thread::yield();
			}
		}
		assert(false);
		return nullptr;
	});

	auto FindAvailableFrameBuffer([&frame_buffer]() -> FrameBuffer* {
		for (;;)
		{
			bool found(false);
			FrameBuffer* t(nullptr);
			{
				std::lock_guard<std::mutex> guard(g_mutex);
				for (auto& w : frame_buffer)
				{
					if (w.associated_worker == nullptr)
					{
						found = true;
						t = std::addressof(w);
					}
				}
			}
			if (found)
				return t;
			else
			{
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(2ms);
				//std::this_thread::yield();
			}
		}
		assert(false);
		return nullptr;
	});

	for (auto &w : workers)
		w.MasterWaitForIdle();

	ScopeTimer timer_("total execution time");
    FPSCounter fps;
	char tmp[256];
	do
	{
		demuxer.Demux(&pVideo, &nVideoBytes);
		dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);

		if (nFrameReturned > 0)
		{
			if (dec.GetOutputFormat() != cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12)
				throw std::runtime_error("Only NV12 format is supported now!");
			if (!inited)
			{
				for (auto& fb : frame_buffer)
					fb.Init(DECODE_BATCH_SIZE, dec.GetFrameSize(), dec.GetOutputFormat(), dec.GetWidth(), dec.GetHeight());
				inited = true;
				cur_frame_buffer->start_frame = nFrame;
			}
			for (;;)
			{
				auto [extra_frames, full] = cur_frame_buffer->Copy(ppFrame, nFrameReturned);
				nFrame += nFrameReturned - extra_frames;
				if (full)
				{
					// send to thread
					auto& worker(*FindIdleWorker());
					worker.SendWork(*cur_frame_buffer);
					// find an available frame buffer
					cur_frame_buffer = FindAvailableFrameBuffer();
					cur_frame_buffer->start_frame = nFrame;
				}
				if (extra_frames)
				{
					// if this happens, full must be true
					assert(full);
					ppFrame += nFrameReturned - extra_frames;
					nFrameReturned -= nFrameReturned - extra_frames;
					continue;
				}
				else
					break;
			}
		}

		//if (nFrameReturned > 0 && nFrame > 20)
		//{
		//	if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
		//	{
		//		std::cout << "cudaVideoSurfaceFormat_YUV444 not supported\n";
		//	}
		//	else
		//	{
		//		{
		//			cudawrapper::CUDADeviceMemoryUnique<std::uint8_t> input_batch(dec.GetFrameSize());
		//			ck2(cuMemcpy(input_batch, (CUdeviceptr)ppFrame[0], dec.GetFrameSize()));
		//			std::int32_t out_width, out_height;
		//			PreprocessNV12(output, output_raw, input_batch, 1, dec.GetFrameSize(), demuxer.GetWidth(), demuxer.GetHeight(), 640, std::addressof(out_width), std::addressof(out_height));

		//			auto u8_buffer(std::make_unique<std::uint8_t[]>(out_width * out_height * 3));
		//			ck2(cuMemcpyDtoH(u8_buffer.get(), output, out_width * out_height * 3 * sizeof(std::uint8_t)));
		//			auto pixel_buffer(std::make_unique<std::uint32_t[]>(out_width * out_height));
		//			for (std::size_t x(0); x < out_width; ++x)
		//				for (std::size_t y(0); y < out_height; ++y)
		//				{
		//					auto r(u8_buffer[out_width * out_height * 0 + out_width * y + x]);
		//					auto g(u8_buffer[out_width * out_height * 1 + out_width * y + x]);
		//					auto b(u8_buffer[out_width * out_height * 2 + out_width * y + x]);
		//					std::uint32_t pixel(
		//						255 << 24 |
		//						(static_cast<std::uint32_t>(b) << 16) |
		//						(static_cast<std::uint32_t>(g) << 8) |
		//						(static_cast<std::uint32_t>(r))
		//					);
		//					pixel_buffer[out_width * y + x] = pixel;
		//				}
		//			lodepng::encode("frame_out.png", reinterpret_cast<std::uint8_t*>(pixel_buffer.get()), out_width, out_height);
		//		}

		//		{
		//			Nv12ToColor32<BGRA32>((uint8_t*)ppFrame[0], dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
		//			auto u32_buffer(std::make_unique<BGRA32[]>(dec.GetWidth() * dec.GetHeight()));
		//			ck2(cuMemcpyDtoH(u32_buffer.get(), dpFrame, dec.GetWidth() * dec.GetHeight() * 4));
		//			auto pixel_buffer(std::make_unique<std::uint32_t[]>(dec.GetWidth() * dec.GetHeight()));
		//			for (std::size_t x(0); x < dec.GetWidth(); ++x)
		//				for (std::size_t y(0); y < dec.GetHeight(); ++y)
		//				{
		//					auto r(u32_buffer[dec.GetWidth() * y + x].c.r);
		//					auto g(u32_buffer[dec.GetWidth() * y + x].c.g);
		//					auto b(u32_buffer[dec.GetWidth() * y + x].c.b);
		//					std::uint32_t pixel(
		//						255 << 24 |
		//						(static_cast<std::uint32_t>(b) << 16) |
		//						(static_cast<std::uint32_t>(g) << 8) |
		//						(static_cast<std::uint32_t>(r))
		//					);
		//					pixel_buffer[dec.GetWidth() * y + x] = pixel;
		//				}
		//			lodepng::encode("frame_ref.png", reinterpret_cast<std::uint8_t*>(pixel_buffer.get()), dec.GetWidth(), dec.GetHeight());
		//		}
		//	}
		//	break;
		//}

        for (int i = 0; i < nFrameReturned; i++)
        {
			//Nv12ToColor32<BGRA32>((uint8_t*)ppFrame[i], dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
			//if (input_batch.empty())
			//	input_batch = std::move(cudawrapper::CUDADeviceMemoryUnique<std::uint8_t>(dec.GetFrameSize()));

			//std::int32_t out_width, out_height;
			//ck2(cuMemcpy(input_batch, (CUdeviceptr)ppFrame[0], dec.GetFrameSize()));
			//PreprocessNV12(output, output_raw, input_batch, 1, dec.GetFrameSize(), demuxer.GetWidth(), demuxer.GetHeight(), 640, std::addressof(out_width), std::addressof(out_height));
			//cuStreamSynchronize(s1);
            //if (dec.GetBitDepth() == 8)
            //{
            //    if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
            //        YUV444ToColor32<BGRA32>((uint8_t*)ppFrame[i], dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
            //    else    // default assumed as NV12
            //        Nv12ToColor32<BGRA32>((uint8_t*)ppFrame[i], dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
            //}
            //else
            //{
            //    if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444_16Bit)
            //        YUV444P16ToColor32<BGRA32>((uint8_t*)ppFrame[i], 2 * dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
            //    else // default assumed as P016
            //        P016ToColor32<BGRA32>((uint8_t*)ppFrame[i], 2 * dec.GetWidth(), (uint8_t*)dpFrame, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
            //}
        }
        
		if (fps.Update(nFrameReturned))
		{
			std::sprintf(tmp, "fps=%.2f",fps.GetFPS());
			std::cout << tmp << "\n";
		}
	} while (nVideoBytes);

	// handle remaining frames
	if (cur_frame_buffer->count)
	{
		auto& worker(*FindIdleWorker());
		worker.SendWork(*cur_frame_buffer);
	}

	{
		std::lock_guard<std::mutex> guard(g_mutex);
		for (auto& w : workers)
			w.NotifyExit();
	}
	for (auto& w : workers)
		w.Join();

	std::size_t frame_ms(static_cast<std::size_t>(static_cast<double>(nFrame) / demuxer.GetFPS() * 1000.0));
	sub_gen.InsertSubtitle(frame_ms, {});

	std::cout << "total frames: " << nFrame << "\n";

	// generate subtitle
	sub_gen.Generate("sanae.srt");

	return 0;
}
