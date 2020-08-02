#pragma once

#include "common.h"

namespace cudawrapper
{
struct CUDAEvent
{
	CUevent event;
	std::uint32_t flags;
	operator CUevent const &() const
	{
		return event;
	}
	bool empty() const
	{
		return event == nullptr;
	}
	CUDAEvent(std::uint32_t flags = CUevent_flags::CU_EVENT_DEFAULT) :event(nullptr), flags(flags)
	{
		ck2(cuEventCreate(std::addressof(event), flags));
	}
	~CUDAEvent() noexcept
	{
		try
		{
			if (event)
				ck2(cuEventDestroy(event));
		} catch (...)
		{

		}
		event = nullptr;
		flags = 0;
	}
	CUDAEvent(CUDAEvent const &other) = delete;
	CUDAEvent &operator=(CUDAEvent const &other) = delete;
	CUDAEvent(CUDAEvent &&other) noexcept
	{
		event = other.event;
		flags = other.flags;
		other.event = nullptr;
		other.flags = 0;
	}
	CUDAEvent &operator=(CUDAEvent &&other) noexcept
	{
		if (std::addressof(other) != this)
		{
			this->~CUDAEvent();
			event = other.event;
			flags = other.flags;
			other.event = nullptr;
			other.flags = 0;
		}
		return *this;
	}
};
}

