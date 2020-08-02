#pragma once

#include "common.h"

namespace cudawrapper
{
	struct CUDAContext
	{
		CUcontext* context;
		int gpu_id;
		std::uint32_t flags;

		operator CUcontext* ()
		{
			return context;
		}

		operator CUcontext const&() const
		{
			return *context;
		}

		bool empty() const
		{
			return context == nullptr;
		}

		CUDAContext() :context(nullptr), gpu_id(0), flags(0)
		{

		}
		CUDAContext(int gpu_id, std::uint32_t flags = 0) :context(nullptr), gpu_id(gpu_id), flags(flags)
		{
			context = new CUcontext;
			if (!context)
				throw std::runtime_error("[*] operator new failed");
			CUdevice device{0};
			ck2(cuDeviceGet(std::addressof(device), gpu_id));
			ck2(cuCtxCreate(context, flags, device));
		}
		~CUDAContext() noexcept
		{
			try
			{
				if (context)
					ck2(cuCtxDestroy(*context));
			} catch (...)
			{

			}
			if (context)
			{
				delete context;
				context = nullptr;
			}
			gpu_id = 0;
			flags = 0;
		}
		CUDAContext(CUDAContext const& other) = delete;
		CUDAContext& operator=(CUDAContext const& other) = delete;

		CUDAContext(CUDAContext&& other) noexcept
		{
			context = other.context;
			gpu_id = other.gpu_id;
			flags = other.flags;
			other.context = nullptr;
			other.gpu_id = 0;
			other.flags = 0;
		}
		CUDAContext& operator=(CUDAContext&& other) noexcept
		{
			if (std::addressof(other) != this)
			{
				this->~CUDAContext();
				context = other.context;
				gpu_id = other.gpu_id;
				flags = other.flags;
				other.context = nullptr;
				other.gpu_id = 0;
				other.flags = 0;
			}
			return *this;
		}
	};
}
