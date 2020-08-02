#pragma once

#include <NvInfer.h>
#include <stdexcept>

namespace cudawrapper
{
struct NvInferContext
{
	nvinfer1::IExecutionContext *context;
	operator nvinfer1::IExecutionContext *() const
	{
		return context;
	}
	NvInferContext(nvinfer1::IExecutionContext *context) :context(context)
	{
		if (!context)
			throw std::runtime_error("Failed to create TensorRT context");
	}

	~NvInferContext() noexcept
	{
		try
		{
			if (context)
				context->destroy();
		} catch (...)
		{

		}
		context = nullptr;
	}
	NvInferContext(NvInferContext const &other) = delete;
	NvInferContext &operator=(NvInferContext const &other) = delete;
	
	NvInferContext(NvInferContext &&other) noexcept = delete;
	NvInferContext &operator=(NvInferContext &&other) noexcept = delete;
};
}


