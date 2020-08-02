#pragma once

#include "common.h"

namespace cudawrapper
{
	template <typename T>
	struct CUDAHostMemoryUnique
	{
		T* ptr;
		std::size_t len;

		bool empty() const
		{
			return ptr == nullptr;
		}

		operator T* () const
		{
			return ptr;
		}

		std::size_t size() const
		{
			return len;
		}

		std::size_t size_bytes() const
		{
			return len * sizeof(T);
		}

		T* at_offset(std::size_t element_size, std::size_t n) const
		{
			return ptr + element_size * n;
		}

		void reallocate(std::size_t n)
		{
			*this = std::move(CUDAHostMemoryUnique<T>(n));
		}

		// you do your job(bounadry check), I am not doing it for you
		T& operator[](std::size_t i)
		{
			return ptr[i];
		}

		T const& operator[](std::size_t i) const
		{
			return ptr[i];
		}

		CUDAHostMemoryUnique() :ptr(nullptr), len(0)
		{

		}
		CUDAHostMemoryUnique(std::size_t l) :ptr(nullptr), len(0)
		{
			ck2(cuMemAllocHost((void**)std::addressof(ptr), l * sizeof(T)));
			len = l;
		}
		~CUDAHostMemoryUnique() noexcept
		{
			try
			{
				if (ptr)
					ck2(cuMemFreeHost(ptr));
			} catch (...)
			{

			}
			ptr = nullptr;
			len = 0;
		}
		CUDAHostMemoryUnique(CUDAHostMemoryUnique<T> const& other)
		{
			if (!other.empty())
			{
				void* tmp;
				ck2(cuMemAllocHost(std::addressof(tmp), other.size()));
				std::memcpy(tmp, other.ptr, other.size());
				ptr = tmp;
				len = other.size();
			}
		}
		CUDAHostMemoryUnique& operator=(CUDAHostMemoryUnique<T> const& other)
		{
			if (std::addressof(other) != this)
			{
				if (!other.empty())
				{
					void* tmp;
					ck2(cuMemAllocHost(std::addressof(tmp), other.size()));
					std::memcpy(tmp, other.ptr, other.size());
					this->~CUDAHostMemoryUnique();
					ptr = tmp;
					len = other.size();
				}
				else
				{
					this->~CUDAHostMemoryUnique();
				}
			}
			return *this;
		}
		CUDAHostMemoryUnique(CUDAHostMemoryUnique<T>&& other) noexcept
		{
			ptr = other.ptr;
			len = other.len;
			other.ptr = nullptr;
			other.len = 0;
		}
		CUDAHostMemoryUnique& operator=(CUDAHostMemoryUnique<T>&& other) noexcept
		{
			if (std::addressof(other) != this)
			{
				this->~CUDAHostMemoryUnique();
				ptr = other.ptr;
				len = other.len;
				other.ptr = nullptr;
				other.len = 0;
			}
			return *this;
		}
	};
}
