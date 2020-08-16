#pragma once

#include "common.h"

#include "CUDAHostMemory.h"

namespace cudawrapper
{
	template <typename T>
	struct CUDADeviceMemoryUnique
	{
		std::size_t len;
		CUdeviceptr ptr;

		operator CUdeviceptr const&() const
		{
			return ptr;
		}

		operator T*() const
		{
			return reinterpret_cast<T*>(ptr);
		}

		std::size_t size() const
		{
			return len;
		}

		std::size_t size_bytes() const
		{
			return len * sizeof(T);
		}

		bool empty() const
		{
			return ptr == 0;
		}

		void reallocate(std::size_t n)
		{
			*this = std::move(CUDADeviceMemoryUnique<T>(n));
		}

		CUdeviceptr at_offset(std::size_t element_size, std::size_t n) const
		{
			return ptr + element_size * n * sizeof(T);
		}

		//void copy_to(CUDADeviceMemoryUnique& dst)
		//{
		//	CUDA_MEMCPY2D desc{};
		//	cuMemcpy2DAsync()
		//}

		void download(CUDAHostMemoryUnique<T>& output, CUstream stream) const
		{
			if (output.empty() || output.size() != size())
				output = std::move(CUDAHostMemoryUnique<T>(size()));
			ck2(cuMemcpyDtoHAsync(output, ptr, size_bytes(), stream));
		}

		void download_block(CUDAHostMemoryUnique<T>& output, CUstream stream = nullptr) const
		{
			if (stream)
			{
				download(output, stream);
				ck2(cuStreamSynchronize(stream));
			}
			else
			{
				if (output.empty() || output.size() != size())
					output = std::move(CUDAHostMemoryUnique<T>(size()));
				ck2(cuMemcpyDtoH(output, ptr, size_bytes()));
			}
		}

		void upload(CUDAHostMemoryUnique<T> const& input, CUstream stream = nullptr)
		{
			if (empty() || size() != input.size())
			{
				this->~CUDADeviceMemoryUnique();
				ck2(cuMemAlloc(std::addressof(ptr), input.size() * sizeof(T)));
				len = input.size();
			}
			ck2(cuMemcpyHtoDAsync(ptr, input, size_bytes(), stream));
		}

		void upload(std::vector<T> const& input, CUstream stream = nullptr)
		{
			if (empty() || size() != input.size())
			{
				this->~CUDADeviceMemoryUnique();
				ck2(cuMemAlloc(std::addressof(ptr), input.size() * sizeof(T)));
				len = input.size();
			}
			ck2(cuMemcpyHtoDAsync(ptr, input.data(), size_bytes(), stream));
		}

		void upload_block(CUDAHostMemoryUnique<T> const& input, CUstream stream = nullptr)
		{
			if (stream)
			{
				upload(input, stream);
				ck2(cuStreamSynchronize(stream));
			}
			else
			{
				if (empty() || size() != input.size())
				{
					this->~CUDADeviceMemoryUnique();
					ck2(cuMemAlloc(std::addressof(ptr), input.size() * sizeof(T)));
					len = input.size();
				}
				ck2(cuMemcpyHtoD(ptr, input, size_bytes()));
			}
		}

		void upload_block(std::vector<T> const& input, CUstream stream = nullptr)
		{
			if (stream)
			{
				upload(input, stream);
				ck2(cuStreamSynchronize(stream));
			}
			else
			{
				if (empty() || size() < input.size())
				{
					this->~CUDADeviceMemoryUnique();
					ck2(cuMemAlloc(std::addressof(ptr), input.size() * sizeof(T)));
					len = input.size();
				}
				ck2(cuMemcpyHtoD(ptr, input.data(), size_bytes()));
			}
		}

		CUDADeviceMemoryUnique() :ptr(0), len(0)
		{

		}

		CUDADeviceMemoryUnique(std::size_t l) :len(l)
		{
			assert(l > 0);
			ck2(cuMemAlloc(std::addressof(ptr), l * sizeof(T)));
		}

		~CUDADeviceMemoryUnique() noexcept
		{
			try
			{
				if (ptr)
					ck2(cuMemFree(ptr));
			} catch (...)
			{
			}
			len = 0;
			ptr = 0;
		}

		[[deprecated("please use explicit copy")]]
		CUDADeviceMemoryUnique(CUDADeviceMemoryUnique<T> const& other) :ptr(0), len(0)
		{
			if (!other.empty())
			{
				CUdeviceptr tmp;
				ck2(cuMemAlloc(&tmp, other.size()));
				ck2(cuMemcpy(tmp, other, other.size()));
				ptr = tmp;
				len = other.size();
			}
		}

		[[deprecated("please use explicit copy")]]
		CUDADeviceMemoryUnique& operator=(CUDADeviceMemoryUnique<T> const& other)
		{
			if (std::addressof(other) != this)
			{
				if (!other.empty())
				{
					CUdeviceptr tmp;
					ck2(cuMemAlloc(&tmp, other.size()));
					ck2(cuMemcpy(tmp, other, other.size()));
					this->~CUDADeviceMemoryUnique();
					ptr = tmp;
					len = other.size();
				}
				else
				{
					this->~CUDADeviceMemoryUnique();
				}
			}
			return *this;
		}
		CUDADeviceMemoryUnique(CUDADeviceMemoryUnique<T>&& other) noexcept
		{
			ptr = other.ptr;
			len = other.len;
			other.ptr = 0;
			other.len = 0;
		}
		CUDADeviceMemoryUnique<T>& operator=(CUDADeviceMemoryUnique<T>&& other) noexcept
		{
			if (std::addressof(other) != this)
			{
				this->~CUDADeviceMemoryUnique();
				ptr = other.ptr;
				len = other.len;
				other.ptr = 0;
				other.len = 0;
			}
			return *this;
		}
	};
}
