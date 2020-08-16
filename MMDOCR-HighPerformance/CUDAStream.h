#pragma once

#include "common.h"

namespace cudawrapper
{
	struct CUDAStream
	{
		CUstream stream;
		std::uint32_t flags;
		operator CUstream const& () const
		{
			return stream;
		}
		bool empty() const
		{
			return stream == nullptr;
		}
		// recommend set flags CUstream_flags::CU_STREAM_NON_BLOCKING
		CUDAStream(std::uint32_t flags = CUstream_flags::CU_STREAM_NON_BLOCKING) :stream(nullptr), flags(flags)
		{
			ck2(cuStreamCreate(std::addressof(stream), flags));
		}
		~CUDAStream() noexcept
		{
			try
			{
				if (stream)
					ck2(cuStreamDestroy(stream));
			} catch (...)
			{

			}
			stream = nullptr;
			flags = 0;
		}
		CUDAStream(CUDAStream const& other) = delete;
		CUDAStream& operator=(CUDAStream const& other) = delete;
		CUDAStream(CUDAStream&& other) noexcept
		{
			stream = other.stream;
			flags = other.flags;
			other.stream = nullptr;
			other.flags = 0;
		}
		CUDAStream& operator=(CUDAStream&& other) noexcept
		{
			if (std::addressof(other) != this)
			{
				this->~CUDAStream();
				stream = other.stream;
				flags = other.flags;
				other.stream = nullptr;
				other.flags = 0;
			}
			return *this;
		}
	};
}
