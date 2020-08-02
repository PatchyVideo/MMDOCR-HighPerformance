#pragma once

#include <NvInfer.h>
#include <stdexcept>

namespace cudawrapper
{
struct NvInferEngine
{
	nvinfer1::ICudaEngine *engine;
	NvInferEngine(nvinfer1::ICudaEngine *engine) :engine(engine)
	{
		if (!engine)
			throw std::runtime_error("Failed to create TensorRT engine");
	}

	auto createContext()
	{
		return engine->createExecutionContext();
	}

	~NvInferEngine() noexcept
	{
		try
		{
			if (engine)
				engine->destroy();
		} catch (...)
		{

		}
		engine = nullptr;
	}
	NvInferEngine(NvInferEngine const &other) = delete;
	NvInferEngine &operator=(NvInferEngine const &other) = delete;
	
	NvInferEngine(NvInferEngine &&other) noexcept = delete;
	NvInferEngine &operator=(NvInferEngine &&other) noexcept = delete;
};
}


