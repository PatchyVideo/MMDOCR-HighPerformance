
#include <queue>

#include "common.h"

/// <summary>
/// Find a min cost assignment using successive shortest path algorithm
/// From https://cp-algorithms.com/graph/min_cost_flow.html
/// </summary>
/// <param name="cost">row major cost matrix</param>
/// <param name="rows">rows</param>
/// <param name="cols">cols</param>
/// <returns>row assignment</returns>
std::vector<int> FindMinCostAssignment(int const* costmat, int rows, int cols)
{
	if (rows == 0 || cols == 0)
	{
		return {};
	}
	struct Edge
	{
		int from, to, capacity, cost;
		Edge(int from, int to, int capacity, int cost) :from(from), to(to), capacity(capacity), cost(cost)
		{

		}
	};
	// step 1: build graph
	std::vector<Edge> edges;
	edges.reserve(rows * cols + rows + cols);
	for (int i(0); i < rows; ++i)
		edges.emplace_back(0, i + 2, 1, 0);
	for (int i(0); i < cols; ++i)
		edges.emplace_back(i + 2 + rows, 1, 1, 0);
	for (int i(0); i < rows; ++i)
		for (int j(0); j < cols; ++j)
			edges.emplace_back(i + 2, j + 2 + rows, 1, costmat[i * cols + j]);
	// step 2: run SSP
	std::vector<std::vector<int>> adj, cost, capacity;
	auto shortest_paths([&adj, &cost, &capacity](int n, int v0, std::vector<int>& d, std::vector<int>& p) {
		d.assign(n, std::numeric_limits<int>::max());
		d[v0] = 0;
		std::vector<char> inq(n, 0);
		std::queue<int> q;
		q.push(v0);
		p.assign(n, -1);

		while (!q.empty())
		{
			int u = q.front();
			q.pop();
			inq[u] = 0;
			for (int v : adj[u])
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
	auto min_cost_flow([&adj, &cost, &capacity, &shortest_paths](int N, std::vector<Edge> edges, int K, int s, int t) {
		adj.assign(N, std::vector<int>());
		cost.assign(N, std::vector<int>(N, 0));
		capacity.assign(N, std::vector<int>(N, 0));
		for (Edge e : edges)
		{
			adj[e.from].push_back(e.to);
			adj[e.to].push_back(e.from);
			cost[e.from][e.to] = e.cost;
			cost[e.to][e.from] = -e.cost;
			capacity[e.from][e.to] = e.capacity;
		}

		int flow = 0;
		int cost = 0;
		std::vector<int> d, p;
		while (flow < K)
		{
			shortest_paths(N, s, d, p);
			if (d[t] == std::numeric_limits<int>::max())
				break;

			// find max flow on that path
			int f = K - flow;
			int cur = t;
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
	auto flow(std::min(rows, cols));
	auto flowcost(min_cost_flow(rows + cols + 2, edges, flow, 0, 1));
	if (flowcost < 0)
		throw std::runtime_error("network flow not satisfied");
	// step 3: find edge with non-zero flow(zero capacity)
	std::vector<int> row_assignment(rows, -1);
	for (int i(0); i < rows; ++i)
	{
		int row(i + 2);
		for (int j(0); j < cols; ++j)
		{
			int col(j + 2 + rows);
			if (capacity[row][col] == 0)
			{
				row_assignment[row - 2] = col - 2 - rows;
				break;
			}
		}
	}
	return row_assignment;
}

