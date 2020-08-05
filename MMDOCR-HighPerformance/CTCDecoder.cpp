
#include <fstream>

#include "CTCDecoder.h"

#include "utf_utils.h"

static std::vector<char32_t> g_alphabet;

void BuildAlphabet()
{
	std::ifstream file("alphabet.txt", std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	auto buffer(std::make_unique<char[]>(size));
	if (file.read(buffer.get(), size))
	{
		char32_t* tmp(new char32_t[size]);
		memset(tmp, 0, size);

		auto num_chars(uu::UtfUtils::SseConvert(reinterpret_cast<uu::UtfUtils::char8_t*>(buffer.get()), reinterpret_cast<uu::UtfUtils::char8_t*>(buffer.get() + size), tmp + 1));
		if (num_chars <= 0)
		{
			throw std::runtime_error("utf8 read failed");
		}
		g_alphabet = std::vector<char32_t>(tmp, tmp + num_chars + 1);
		std::cout << "Alphabet read, total " << static_cast<std::size_t>(num_chars) << " chars\n";

		delete[] tmp;
	}
	else
		throw std::runtime_error("failed to read alphabet file");
}

std::int32_t GetSpaceWidth(std::int32_t const *seq, std::size_t seq_len)
{
	std::int32_t spcace_between_chars(0), num_chars(0), num_space_since_last_char(0);
	enum class State
	{
		Init,
		Char,
		Space,
		SpaceAfterChar
	} state(State::Init);
	std::int32_t last_char_index(-1);
	for (std::size_t i(0); i < seq_len; ++i)
	{
		bool isSpace(seq[i] == 0);
		switch (state)
		{
		case State::Init:
			if (isSpace)
				state = State::Space;
			else
			{
				state = State::Char;
				last_char_index = seq[i];
				++num_chars;
			}
			break;
		case State::Char:
			if (isSpace)
			{
				state = State::SpaceAfterChar;
				num_space_since_last_char = 1;
			} else
			{
				state = State::Char;
				if (seq[i] != last_char_index)
				{
					last_char_index = seq[i];
					++num_chars;
				}
			}
			break;
		case State::Space:
			if (isSpace)
			{

			} else
			{
				state = State::Char;
				last_char_index = seq[i];
				++num_chars;
			}
			break;
		case State::SpaceAfterChar:
			if (isSpace)
			{
				++num_space_since_last_char;
			} else
			{
				state = State::Char;
				spcace_between_chars += num_space_since_last_char;
				num_space_since_last_char = 0;
				last_char_index = seq[i];
				++num_chars;
			}
			break;
		default:
			break;
		}
	}
	if (num_chars == 0)
		return std::numeric_limits<std::int32_t>::max();
	return static_cast<std::int32_t>(static_cast<float>(spcace_between_chars) / static_cast<float>(num_chars));
}

std::vector<std::u32string> CTCDecode(cudawrapper::CUDAHostMemoryUnique<std::int32_t> const& ocr_result, std::size_t num_imgs, std::size_t image_width)
{
	std::vector<std::u32string> result(num_imgs);
	for (std::int64_t i(0); i < num_imgs; ++i)
	{
		std::u32string cur_string;
		std::vector<std::int32_t> t(image_width);
		std::int32_t char_index(ocr_result.at_offset(image_width, i)[0]);
		//auto space_width(GetSpaceWidth(ocr_result.at_offset(image_width, i), image_width));
		t[0] = char_index;
		if (char_index != 0)
		{
			cur_string.append(1, g_alphabet[char_index]);
		}
		std::int32_t consecutive_zeros(0);
		for (std::size_t j(1); j < image_width; ++j)
		{
			std::int32_t char_index(ocr_result.at_offset(image_width, i)[j]);
			//cur_string.append(1, char_index!=0?g_alphabet[char_index]:' '); continue;
			t[j] = char_index;
			if (char_index != 0 && char_index != t[j - 1])
			{
				cur_string.append(1, g_alphabet[char_index]);
				consecutive_zeros = 0;
			}/* else if (char_index == 0)
			{
				++consecutive_zeros;
				if (consecutive_zeros > space_width)
				{
					cur_string.append(1, ' ');
					consecutive_zeros = 0;
				}
			}*/
		}
		result[i] = cur_string;
	}
	return result;
}
