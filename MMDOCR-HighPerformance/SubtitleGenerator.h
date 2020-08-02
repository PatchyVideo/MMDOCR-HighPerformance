#pragma once

#include "common.h"

void BuildSpaceCharacterU32();

struct SubtitleGenerator
{
	void InsertSubtitle(std::size_t time_ms, std::vector<std::u32string> const &texts);
	void Generate(std::string_view output_filename);

	float levenshtein_threshold;
	std::size_t subtitle_duration_threshold;
	std::vector<std::pair<std::size_t, std::vector<std::u32string>>> buffer;

	SubtitleGenerator(float levenshtein_threshold = 0.8f, std::size_t subtitle_duration_threshold = 100) :levenshtein_threshold(levenshtein_threshold), subtitle_duration_threshold(subtitle_duration_threshold)
	{

	}
};
