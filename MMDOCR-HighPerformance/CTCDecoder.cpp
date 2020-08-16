
#include <fstream>
#include <queue>
#include <algorithm>

#include "CTCDecoder.h"

#include "utf_utils.h"

static std::vector<char32_t> g_alphabet;
static std::unordered_map<std::pair<char32_t, char32_t>, float, pair_hash> g_bigram_probs;

void BuildAlphabet()
{
	std::ifstream file("alphabet.txt", std::ios::binary | std::ios::ate);
	if (file.bad() || file.fail())
		throw std::runtime_error("failed to open alphabet.txt");
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

void BuildBigramProbs()
{
	std::ifstream ifs("bigram_probs_v2.txt");
	if(ifs.bad() || ifs.fail())
		throw std::runtime_error("failed to open bigram_probs_v2.txt");
	std::string char_pair; float prob;
	while (ifs)
	{
		ifs >> char_pair >> prob;
		std::u32string char_pair_u32(ConvertU8toU32(char_pair));
		if (char_pair_u32.size() != 2)
			throw std::runtime_error("corrupted bigram_probs_v2.txt");
		g_bigram_probs[{char_pair_u32[0], char_pair_u32[1]}] = prob;
	}
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

inline float negative_log_probability(float prob) { return -std::logf(prob); }

inline float transition_negative_log_probability(char32_t a, char32_t b) // -logP(b|a)
{
	std::pair<char32_t, char32_t> item{ a, b };
	if (g_bigram_probs.count(item))
		return negative_log_probability(g_bigram_probs[item]);
	else
		return 10000000.0f;
	//if (a == 0 || b == 0)
	//	return 10000.0f;
	//if (a == 'I' && b == '\'')
	//	return 0.0f;
	//return 10.0f;
}


std::vector<std::pair<int, float>> list_source_vertices(std::vector<std::vector<std::pair<char32_t, float>>> const& candidates, int start_vertex_id, int j, std::size_t k)
{
	if (j == 0)
		return { { start_vertex_id, 0 } };
	else
	{
		std::vector<std::pair<int, float>> result;
		for (std::size_t u(0); u < candidates[j - 1].size(); ++u)
		{
			if (candidates[j - 1][u].first == 0)
			{
				auto recursive_vertices(list_source_vertices(candidates, start_vertex_id, j - 1, k));
				for (auto [ver_id, neglogprob] : recursive_vertices)
					result.emplace_back(ver_id, neglogprob + candidates[j - 1][u].second * 10.0f);
			}
			else
				result.emplace_back((j - 1) * k * 2 + u * 2 + 1, 0);
		}
		return result;
	}
}

std::u32string DecodeCandidates(std::vector<std::vector<std::pair<char32_t, float>>> const& candidates, std::int64_t k, float cost_scale = 1000.0f)
{
	if (candidates.size() == 0)
		return {};
	if (candidates.size() == 1)
		return std::u32string({ candidates[0][0].first });
	
	// build graph
	struct Edge
	{
		std::int64_t from, to, capacity, cost;
		Edge(std::int64_t from, std::int64_t to, std::int64_t capacity, float cost) :from(from), to(to), capacity(capacity), cost(static_cast<std::int64_t>(cost))
		{

		}
	};

	std::int64_t start_vertex_id(candidates.size() * k * 2);
	std::int64_t end_vertext_id(start_vertex_id + 1);

	std::vector<Edge> edges;
	std::vector<std::vector<std::int64_t>> next(end_vertext_id + 1);

	std::int64_t num_vertices(2);

	for (std::size_t u(0); u < candidates.front().size(); ++u)
	{
		edges.emplace_back(0 * k * 2 + u * 2, 0 * k * 2 + u * 2 + 1, 1, candidates.front()[0].second * cost_scale); // first column
		next[0 * k * 2 + u * 2].emplace_back(0 * k * 2 + u * 2 + 1);
		edges.emplace_back(start_vertex_id, 0 * k * 2 + u * 2, 1, 0.0f);
		next[start_vertex_id].emplace_back(0 * k * 2 + u * 2);
		num_vertices += 2;
	}

	auto get_vertex_char([&candidates, k](std::int64_t v) -> char32_t {
		std::int64_t col(v / (k * 2));
		std::int64_t row(v % (k * 2));
		row -= row & 1;
		return candidates[col][row / 2].first;
	});

	for (std::size_t i(1); i < candidates.size(); ++i)
	{
		auto src_candidates(list_source_vertices(candidates, start_vertex_id, i, k));
		auto dst_candidates(candidates[i]);
		for (std::size_t v(0); v < dst_candidates.size(); ++v)
		{
			edges.emplace_back(i * k * 2 + v * 2, i * k * 2 + v * 2 + 1, 1, candidates[i][v].second * cost_scale); // column i
			next[i * k * 2 + v * 2].emplace_back(i * k * 2 + v * 2 + 1);
			num_vertices += 2;
			for (std::size_t u(0); u < src_candidates.size(); ++u)
			{
				if (src_candidates[u].first == start_vertex_id) {
					edges.emplace_back(start_vertex_id, i * k * 2 + v * 2, 1, src_candidates[u].second * cost_scale);
					next[start_vertex_id].emplace_back(i * k * 2 + v * 2);
				}
				else {
					edges.emplace_back(src_candidates[u].first, i * k * 2 + v * 2, 1, (src_candidates[u].second + transition_negative_log_probability(get_vertex_char(src_candidates[u].first), candidates[i][v].first)) * cost_scale);
					next[src_candidates[u].first].emplace_back(i * k * 2 + v * 2);
				}
			}
		}
	}

	for (auto [src_id, addtional_cost] : list_source_vertices(candidates, start_vertex_id, candidates.size(), k)) {
		edges.emplace_back(src_id, end_vertext_id, 1, addtional_cost * cost_scale);
		next[src_id].emplace_back(end_vertext_id);
	}

	// step 2: run SSP
	std::vector<std::vector<std::int64_t>> adj, cost, capacity;
	auto shortest_paths([&adj, &cost, &capacity](std::int64_t n, std::int64_t v0, std::vector<std::int64_t>& d, std::vector<std::int64_t>& p) {
		d.assign(n, std::numeric_limits<std::int64_t>::max());
		d[v0] = 0;
		std::vector<char> inq(n, 0);
		std::queue<std::int64_t> q;
		q.push(v0);
		p.assign(n, -1);

		while (!q.empty())
		{
			std::int64_t u = q.front();
			q.pop();
			inq[u] = 0;
			for (std::int64_t v : adj[u])
			{
				if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v])
				{
					d[v] = d[u] + cost[u][v];
					p[v] = u;
					if (!inq[v])
					{
						inq[v] = 1;
						q.push(v);
					}
				}
			}
		}
	});
	auto min_cost_flow([&adj, &cost, &capacity, &shortest_paths](std::int64_t N, std::vector<Edge> edges, std::int64_t K, std::int64_t s, std::int64_t t) -> std::int64_t {
		adj.assign(N, std::vector<std::int64_t>());
		cost.assign(N, std::vector<std::int64_t>(N, 0));
		capacity.assign(N, std::vector<std::int64_t>(N, 0));
		for (Edge e : edges)
		{
			adj[e.from].push_back(e.to);
			adj[e.to].push_back(e.from);
			cost[e.from][e.to] = e.cost;
			cost[e.to][e.from] = -e.cost;
			capacity[e.from][e.to] = e.capacity;
		}

		std::int64_t flow = 0;
		std::int64_t cost = 0;
		std::vector<std::int64_t> d, p;
		while (flow < K)
		{
			shortest_paths(N, s, d, p);
			if (d[t] == std::numeric_limits<std::int64_t>::max())
				break;

			// find max flow on that path
			std::int64_t f = K - flow;
			std::int64_t cur = t;
			while (cur != s)
			{
				f = std::min(f, capacity[p[cur]][cur]);
				cur = p[cur];
			}

			// apply flow
			flow += f;
			cost += f * d[t];
			cur = t;
			while (cur != s)
			{
				capacity[p[cur]][cur] -= f;
				capacity[cur][p[cur]] += f;
				cur = p[cur];
			}
		}

		if (flow < K)
			return -1;
		else
			return cost;
	});
	auto flowcost(min_cost_flow(end_vertext_id + 1, edges, num_vertices, start_vertex_id, end_vertext_id)); // always -1

	std::u32string result;

	std::int64_t cur(start_vertex_id);
	while (next[cur].size())
	{
		bool found(false);
		for (auto next_id : next[cur])
		{
			if (capacity[cur][next_id] == 0)
			{

				if (next_id % 2 == 0)
					result.append(1, get_vertex_char(next_id));
				cur = next_id;
				found = true;
				break;
			}
		}
		if (!found)
			throw std::runtime_error("flow failed");
	}
	return result;
}

std::u32string DecodeSingleSentenceTop1(std::int32_t const* const indices, float const* const probs, std::size_t k, std::size_t sentence_length, float threshold = 0.9f)
{
	std::u32string result;
	char32_t last_ch(std::numeric_limits<char32_t>::max());
	for (std::size_t i(0); i < sentence_length; ++i)
	{
		char32_t top1_ch(g_alphabet[indices[i * k + 0]]);
		float top1_prob(probs[i * k + 0]);
		if (top1_ch == last_ch || top1_ch == 0) {
			last_ch = top1_ch;
			continue;
		}
		last_ch = top1_ch;
		result.append(1, top1_ch);
	}
	return result;
}

std::u32string DecodeSingleSentence(std::int32_t const* const indices, float const* const probs, std::size_t k, std::size_t sentence_length, float threshold = 0.9f)
{
	// generate candidates
	std::vector<std::vector<std::pair<char32_t, float>>> candidates;
	char32_t last_ch(std::numeric_limits<char32_t>::max());
	for (std::size_t i(0); i < sentence_length; ++i)
	{
		char32_t top1_ch(g_alphabet[indices[i * k + 0]]);
		float top1_prob(probs[i * k + 0]);
		if (top1_ch == last_ch) {
			last_ch = top1_ch;
			continue;
		}
		last_ch = top1_ch;
		if (top1_prob >= threshold && top1_ch == 0)
			continue; // we are certrain this is [blank], skipping
		if (top1_prob < threshold) {
			std::vector<std::pair<char32_t, float>> cur;
			for (std::size_t j(0); j < k; ++j) {
				char32_t topk_ch(g_alphabet[indices[i * k + j]]);
				float topk_prob(probs[i * k + j]);
				cur.emplace_back(topk_ch, negative_log_probability(topk_prob));
			}
			candidates.emplace_back(cur);
		}
		else {
			std::vector<std::pair<char32_t, float>> cur;
			cur.emplace_back(top1_ch, negative_log_probability(top1_prob));
			candidates.emplace_back(cur);
		}
	}

	return DecodeCandidates(candidates, k);
}

std::vector<std::u32string> CTCDecode(
	cudawrapper::CUDAHostMemoryUnique<std::int32_t> const& ocr_result_indices,
	cudawrapper::CUDAHostMemoryUnique<float> const& ocr_result_probs,
	std::size_t num_imgs,
	std::size_t image_width,
	std::size_t k
)
{
	std::vector<std::u32string> result(num_imgs);
	for (std::int64_t i(0); i < num_imgs; ++i)
	{
		result[i] = DecodeSingleSentenceTop1(ocr_result_indices.at_offset(image_width * k, i), ocr_result_probs.at_offset(image_width * k, i), k, image_width);
	}
	return result;
}
