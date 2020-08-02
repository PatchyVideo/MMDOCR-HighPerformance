#pragma once

#include <numeric>

#include "common.h"

std::vector<BBox> MergeBBoxes(std::vector<BBox> const& detects)
{
	std::vector<BBox> result;
	std::uint32_t num_detects(detects.size());
	result.reserve(num_detects);

	std::vector<std::uint32_t> parents(num_detects);
	std::iota(parents.begin(), parents.end(), 0);

	auto find([&parents](std::uint32_t a) {
		while (parents[a] != a)
		{
			parents[a] = parents[parents[a]];
			a = parents[a];
		}
		return a;
			  });
	auto unite([&parents, &find](std::uint32_t a, std::uint32_t b) {
		parents[find(a)] = find(b);
			   });

	auto can_merge([](BBox const& a, BBox const& b) {
		if (a.contains(b) || b.contains(a))
			return true;
		auto char_size(std::min(a.height, b.height));
		if (std::abs((a.y + a.height / 2) - (b.y + b.height / 2)) * 1.5 > char_size)
		{
			if (std::abs(a.height - b.height) > char_size)
				return false;
			if (std::abs(a.y - b.y) * 2 > char_size)
				return false;
		}
		if (a.x < b.x)
		{
			if (std::abs(a.right() - b.x) > char_size)
				return false;
			else
				return true;
		}
		else
		{
			if (std::abs(b.right() - a.x) > char_size)
				return false;
			else
				return true;
		}
		return false;
				   });

	for (std::uint32_t i(0); i < num_detects; ++i)
		for (std::uint32_t j(i + 1); j < num_detects; ++j)
		{
			if (can_merge(detects[i], detects[j]))
				unite(i, j);
		}

	bool compression(true);
	while (compression)
	{
		compression = false;
		for (std::uint32_t i(0); i < num_detects; ++i)
		{
			auto root(i);
			while (parents[root] != parents[parents[root]])
			{
				parents[root] = parents[parents[root]];
				root = parents[root];
				compression = true;
			}
		}
	}

	std::unordered_map<std::uint32_t, BBox> m;
	for (std::uint32_t cur(0); cur < num_detects; ++cur)
	{
		auto root(parents[cur]);
		if (m.count(root))
			m[root].merge(detects[cur]);
		else
			m[root] = detects[cur];
	}
	for (auto const& item : m)
		result.emplace_back(item.second);

	return result;
}
