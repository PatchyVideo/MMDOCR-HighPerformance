#pragma once

#include "common.h"

namespace cudawrapper
{
struct CUDAThreadContext
{
	CUcontext ctx;
	CUDAThreadContext(CUcontext ctx) :ctx(ctx)
	{
		ck2(cuCtxPushCurrent(ctx));
	}
	~CUDAThreadContext() noexcept
	{
		try
		{
			ck2(cuCtxPopCurrent(std::addressof(ctx)));
		} catch (...)
		{

		}
	}
	CUDAThreadContext(CUDAThreadContext const &a) = delete;
	CUDAThreadContext &operator=(CUDAThreadContext const &a) = delete;
	CUDAThreadContext(CUDAThreadContext &&a) = delete;
	CUDAThreadContext &operator=(CUDAThreadContext &&a) = delete;
};
}
