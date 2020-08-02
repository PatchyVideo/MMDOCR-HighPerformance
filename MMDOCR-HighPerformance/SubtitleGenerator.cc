
#include <algorithm>
#include <numeric>
#include <fstream>

#include "utf_utils.h"
#include "SubtitleGenerator.h"

std::vector<char32_t> g_spaces;

void BuildSpaceCharacterU32()
{
	std::string const spaces("○●°•○○ ¤《》¡¿':.\n\r[] \t\v\f{}-_■=+`~!@#$%^&*();'\", <> / ? \\ | －＞＜。，《》【】　？！￥…（）、：；·「」『』〔〕［］｛｝｟｠〉〈〖〗〘〙〚〛゠＝‥※＊〽〓〇＂“”‘’＃＄％＆＇＋．／＠＼＾＿｀｜～｡｢｣､･ｰﾟ￠￡￢￣￤￨￩￪￫￬￭￮・◆◊→←↑↓↔—'");
	char32_t *tmp(new char32_t[spaces.size()]);
	memset(tmp, 0, spaces.size());

	auto num_chars(uu::UtfUtils::SseConvert(reinterpret_cast<unsigned char const *>(&*spaces.cbegin()), reinterpret_cast<unsigned char const *>(&*spaces.cend()), tmp));
	if (num_chars <= 0)
	{
		throw std::runtime_error("utf8 read failed");
	}
	g_spaces = std::vector<char32_t>(tmp, tmp + num_chars);
}

std::vector<std::u32string> FilterSpaceOnlyString(std::vector<std::u32string> const &raw)
{
	std::vector<std::u32string> result;
	result.reserve(raw.size());
	for (auto const &s : raw)
	{
		std::uint32_t num_allowed_chars(0);
		for (auto const& cdpt : s)
		{
			if (std::find(g_spaces.cbegin(), g_spaces.cend(), cdpt) == g_spaces.cend())
				++num_allowed_chars;
		}
		if (num_allowed_chars)
			result.emplace_back(s);
	}
	return result;
}

std::vector<std::u32string> FilterSpaceCharacters(std::vector<std::u32string> const &raw)
{
	std::vector<std::u32string> result;
	result.reserve(raw.size());
	for (auto const &s : raw)
	{
		std::vector<char32_t> allowed_chars;
		allowed_chars.reserve(s.size());
		std::copy_if(s.cbegin(), s.cend(), std::back_inserter(allowed_chars), [](char32_t cdpt) {
			return std::find(g_spaces.cbegin(), g_spaces.cend(), cdpt) == g_spaces.cend();
		});
		if (allowed_chars.size())
			result.emplace_back(allowed_chars.cbegin(), allowed_chars.cend());
	}
	return result;
}

// from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C++
float EditDistanceRatio(const std::u32string &s1, const std::u32string &s2)
{
	std::size_t const len1 = s1.size(), len2 = s2.size();
	std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

	d[0][0] = 0;
	for (unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
	for (unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

	for (unsigned int i = 1; i <= len1; ++i)
		for (unsigned int j = 1; j <= len2; ++j)
			d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
	return (static_cast<float>(len1 + len2) - static_cast<float>(d[len1][len2])) / static_cast<float>(len1 + len2);
}

struct SubtitleSegment
{
	std::vector<std::u32string> texts, filtered_texts;
	std::size_t start, end;

	SubtitleSegment(std::vector<std::u32string> texts, std::size_t start, std::size_t end) :texts(texts), start(start), end(end)
	{
		filtered_texts = FilterSpaceCharacters(texts);
	}

	SubtitleSegment() :start(0), end(0)
	{

	}

	std::size_t duration() const noexcept
	{
		return end - start;
	}
};

std::vector<int> GetSegmentsAssignment(SubtitleSegment const &a, SubtitleSegment const &b, float cost_scale = 1000.0f)
{
	auto M(a.filtered_texts.size()), N(b.filtered_texts.size());
	assert(M == N);
	auto costmat(std::make_unique<int[]>(M * N));

	for (std::size_t i(0); i < M; ++i)
		for (std::size_t j(0); j < N; ++j)
		{
			costmat[i * N + j] = static_cast<int>((1.0f - EditDistanceRatio(a.filtered_texts[i], b.filtered_texts[j])) * cost_scale);
		}
	return FindMinCostAssignment(costmat.get(), M, N);
}

SubtitleSegment MergeSubtitleGroup(std::vector<SubtitleSegment> segs)
{
	std::sort(segs.begin(), segs.end(), [](auto const &a, auto const &b) {return a.start < b.start; });

	std::vector<std::vector<int>> assignments(segs.size() - 1, std::vector<int>());
	for (std::size_t i(1); i < segs.size(); ++i)
		assignments[i - 1] = GetSegmentsAssignment(segs[i - 1], segs[i]);

	std::vector<std::u32string> result_texts;
	for (auto const &[txt_idx, txt] : enumerate(segs.front().texts))
	{
		std::unordered_map<std::u32string, std::size_t> text_duration_map;
		text_duration_map[txt] = segs.front().duration();
		auto cur_txt_idx(txt_idx);
		for (std::size_t i(1); i < segs.size(); ++i)
		{
			cur_txt_idx = assignments[i - 1][cur_txt_idx];
			auto const &cur_text(segs[i].texts[cur_txt_idx]);
			text_duration_map[cur_text] += segs[i].duration();
		}
		std::u32string longest_text;
		std::size_t longest_duration(0);
		for (auto const &[cur_text, cur_duration] : text_duration_map)
			if (cur_duration > longest_duration)
			{
				longest_duration = cur_duration;
				longest_text = cur_text;
			}
		result_texts.emplace_back(longest_text);
	}

	return SubtitleSegment(result_texts, segs.front().start, segs.back().end);
}

std::vector<SubtitleSegment> MergeSubtitleSegments(std::vector<SubtitleSegment> const &segs, float levenshtein_threshold)
{
	std::vector<std::uint32_t> parents(segs.size());
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

	auto can_merge([&segs, levenshtein_threshold](std::size_t i, std::size_t j) {
		if (segs[i].filtered_texts.size() != segs[j].filtered_texts.size())
			return false;
		auto const &row_assignments(GetSegmentsAssignment(segs[i], segs[j]));
		for (std::size_t a(0); a < row_assignments.size(); ++a)
		{
			auto b(row_assignments[a]);
			if (EditDistanceRatio(segs[i].filtered_texts[a], segs[j].filtered_texts[b]) < levenshtein_threshold)
				return false;
		}
		return true;
	});

	for (std::size_t i(1); i < segs.size(); ++i)
	{
		if (can_merge(i - 1, i))
			unite(i - 1, i);
	}

	bool compression(true);
	while (compression)
	{
		compression = false;
		for (std::uint32_t i(0); i < segs.size(); ++i)
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

	std::unordered_map<std::uint32_t, std::vector<SubtitleSegment>> segment_groups;
	for (std::uint32_t cur(0); cur < segs.size(); ++cur)
	{
		auto root(find(cur));
		segment_groups[root].emplace_back(segs[cur]);
	}
	std::vector<SubtitleSegment> result;
	for (auto const &[root, item] : segment_groups)
		result.emplace_back(MergeSubtitleGroup(item));
	std::sort(result.begin(), result.end(), [](auto const &a, auto const &b) {return a.start < b.start; });
	return result;
}


void SubtitleGenerator::InsertSubtitle(std::size_t time_ms, std::vector<std::u32string> const &texts)
{
	buffer.emplace_back(time_ms, FilterSpaceOnlyString(texts));
}

void SubtitleGenerator::Generate(std::string_view output_filename)
{
	std::vector<SubtitleSegment> segments;
	std::sort(buffer.begin(), buffer.end(), [](auto const &a, auto const &b) {return a.first < b.first; });
	segments.emplace_back(std::vector<std::u32string>{}, 0, buffer[0].first);
	for (std::size_t i(1); i < buffer.size(); ++i)
	{
		segments.emplace_back(buffer[i - 1].second, buffer[i - 1].first, buffer[i].first);
	}
	auto merged_stage1(MergeSubtitleSegments(segments, levenshtein_threshold));

	decltype(segments) filtered_stage2;
	std::copy_if(merged_stage1.cbegin(), merged_stage1.cend(), std::back_inserter(filtered_stage2), [this](SubtitleSegment const &s) {return s.duration() > this->subtitle_duration_threshold; });

	auto result(MergeSubtitleSegments(filtered_stage2, levenshtein_threshold));

	std::ofstream ofs(output_filename.data());

	std::size_t text_counter(1);
	for (auto const &seg : result)
	{
		if (seg.texts.size() == 0)
			continue;
		ofs << text_counter << "\n";
		ofs << mil2str(seg.start) << " --> " << mil2str(seg.end) << "\n";
		for (auto const &s : seg.texts)
		{
			auto su8(ConvertU32toU8(s));
			ofs << su8 << "\n";
		}
		ofs << "\n";
		++text_counter;
	}
}
