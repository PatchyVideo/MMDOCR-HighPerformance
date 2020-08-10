#pragma once

// CUDA Driver API
#include <cuda.h>
// CUDA Runtime API
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>

#include <utility>
#include <algorithm>
#include <memory>
#include <vector>

#include <exception>
#include <stdexcept>
#include <cassert>

#include <iostream>

#include "utf_utils.h"

//#include <thrust/host_vector.h>

inline void ThrowIfFailed(CUresult const& ret, int line, char const* filename)
{
	if (ret != CUresult::CUDA_SUCCESS)
	{
		char const* desc{ nullptr }, * name{ nullptr };
		cuGetErrorName(ret, std::addressof(name));
		cuGetErrorString(ret, std::addressof(desc));
		std::cout << "[*] CUDA Driver Error at file " << filename << " line " << line << "\n";
		if (desc && name)
		{
			std::cout << "[*] " << name << "\n";
			std::cout << "[*] " << desc << "\n";
		}
		else
		{
			std::cout << "[*] Error acquiring description for this error\n";
		}
		throw std::runtime_error("CUDA Driver Error");
	}
}

inline void ThrowIfFailed(cudaError_t const& ret, int line, char const* filename)
{
	if (ret != cudaError_t::cudaSuccess)
	{
		char const* desc{ cudaGetErrorString(ret) }, * name{ cudaGetErrorName(ret) };
		std::cout << "[*] CUDA Runtime Error at file " << filename << " line " << line << "\n";
		if (desc && name)
		{
			std::cout << "[*] " << name << "\n";
			std::cout << "[*] " << desc << "\n";
		}
		else
		{
			std::cout << "[*] Error acquiring description for this error\n";
		}
		throw std::runtime_error("CUDA Runtime Error");
	}
}

#define ck2(call) ThrowIfFailed(call, __LINE__, __FILE__)

// from http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
template <typename T,
	typename TIter = decltype(std::begin(std::declval<T>())),
	typename = decltype(std::end(std::declval<T>()))>
	constexpr auto enumerate(T&& iterable)
{
	struct iterator
	{
		size_t i;
		TIter iter;
		bool operator != (const iterator& other) const { return iter != other.iter; }
		void operator ++ () { ++i; ++iter; }
		auto operator * () const { return std::tie(i, *iter); }
	};
	struct iterable_wrapper
	{
		T iterable;
		auto begin() { return iterator{ 0, std::begin(iterable) }; }
		auto end() { return iterator{ 0, std::end(iterable) }; }
	};
	return iterable_wrapper{ std::forward<T>(iterable) };
}

struct BBox
{
	std::int32_t x, y, width, height;
	BBox() noexcept :x(0), y(0), width(0), height(0) {}
	BBox(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height) noexcept :x(x), y(y), width(width), height(height) {}
	auto left() const noexcept { return x; }
	auto top() const noexcept { return y; }
	auto right() const noexcept { return x + width; }
	auto bottom() const noexcept { return y + height; }
	cv::Rect rect() const noexcept { return cv::Rect(x, y, width, height); }
	void crop(std::int32_t max_width, std::int32_t max_height) noexcept
	{
		x = std::max(x, 0);
		y = std::max(y, 0);
		width = std::min(width, max_width - x);
		height = std::min(height, max_height - y);
	}
	void merge(BBox const& other) noexcept
	{
		auto left1(left()), top1(top()), right1(right()), bottom1(bottom());
		auto left2(other.left()), top2(other.top()), right2(other.right()), bottom2(other.bottom());
		auto left3(std::min(left1, left2)), top3(std::min(top1, top2)), right3(std::max(right1, right2)), bottom3(std::max(bottom1, bottom2));
		x = left3;
		y = top3;
		width = right3 - left3;
		height = bottom3 - top3;
	}
	bool contains(BBox const& a) const noexcept
	{
		return x <= a.x && y <= a.y && right() >= a.right() && bottom() >= a.bottom();
	}
	void scale(float scale) noexcept
	{
		x = static_cast<std::int32_t>(scale * static_cast<float>(x));
		y = static_cast<std::int32_t>(scale * static_cast<float>(y));
		width = static_cast<std::int32_t>(scale * static_cast<float>(width));
		height = static_cast<std::int32_t>(scale * static_cast<float>(height));
	}
	float IoU(BBox const& other) const noexcept
	{
		auto xA(std::max(left(), other.left()));
		auto yA(std::max(top(), other.top()));
		auto xB(std::min(right(), other.right()));
		auto yB(std::min(bottom(), other.bottom()));

		auto interArea(std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1));

		auto boxAArea(width * height);
		auto boxBArea(other.width * other.height);

		auto iou(float(interArea) / float(boxAArea + boxBArea - interArea));
		return std::max(0.0f, std::min(iou, 1.0f));
	}
};

class ScopeTimer
{
	std::string_view s;
	std::chrono::high_resolution_clock::time_point t0;
public:
	ScopeTimer(std::string_view s) :s(s), t0(std::chrono::high_resolution_clock::now()) {}
	ScopeTimer(const ScopeTimer&) = delete;
	ScopeTimer& operator=(const ScopeTimer&) = delete;
	~ScopeTimer()
	{
		std::chrono::duration<float> diff(std::chrono::high_resolution_clock::now() - t0);
		std::cout << s << " " << diff.count() << "s\n";
	}
};

struct pair_hash
{
	template <class T1, class T2>
	std::size_t operator () (const std::pair<T1, T2>& p) const
	{
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);

		// Mainly for demonstration purposes, i.e. works but is overly simple
		// In the real world, use sth. like boost.hash_combine
		h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
		return h1;
	}
};

using subtitle_index = std::pair<std::uint32_t, std::uint32_t>;
using comparison_pair = std::pair<std::uint32_t, std::uint32_t>;

std::vector<int> FindMinCostAssignment(int const* costmat, int rows, int cols);

inline std::string mil2str(std::size_t ms)
{
	int sec(ms / 1000);
	int remain_ms(ms - sec * 1000);
	int minutes = sec / 60;
	int remain_sec = sec - minutes * 60;
	int hrs = minutes / 60;
	int remain_minutes = minutes - hrs * 60;
	char tmp[64];
	std::sprintf(tmp, "%02d:%02d:%02d.%03d", hrs, remain_minutes, remain_sec, remain_ms);
	return std::string(tmp);
}

inline std::string ConvertU32toU8(std::u32string const &s)
{
	uu::UtfUtils::char8_t *tmp(new uu::UtfUtils::char8_t[s.size() * 4]);
	uu::UtfUtils::char8_t *pos = tmp;
	for (auto cdpt : s)
	{
		uu::UtfUtils::GetCodeUnits(cdpt, pos);
	}
	auto ret = std::string(tmp, pos);
	delete[] tmp;
	return ret;
}


inline std::u32string ConvertU8toU32(std::string const& s)
{
	auto tmp(std::make_unique<char32_t[]>(s.size() + 1));
	memset(tmp.get(), 0, s.size() + 1);

	auto num_chars(uu::UtfUtils::SseConvert(reinterpret_cast<unsigned char const*>(s.data()), reinterpret_cast<unsigned char const*>(s.data() + s.size()), tmp.get()));
	if (num_chars <= 0)
	{
		throw std::runtime_error("utf8 read failed");
	}
	return std::u32string(tmp.get(), tmp.get() + num_chars);
}

