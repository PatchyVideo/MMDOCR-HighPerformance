#pragma once

#include <chrono>

struct FPSCounter
{
	std::size_t frame_counter;
	std::size_t frame_since_last_update;
	std::chrono::high_resolution_clock::time_point last_time;
	float fps;
	FPSCounter() :frame_counter(0), frame_since_last_update(0), last_time(std::chrono::high_resolution_clock::now()), fps(0.0f)
	{

	}
	bool Update(std::size_t n_frame = 1)
	{
		frame_counter += n_frame;
		frame_since_last_update += n_frame;
		auto now(std::chrono::high_resolution_clock::now());
		auto elpased(std::chrono::duration_cast<std::chrono::microseconds>(now - last_time));
		if (elpased.count() > 1000000)
		{
			fps = 1000000.0f * static_cast<float>(frame_since_last_update) / static_cast<float>(elpased.count());

			last_time = std::chrono::high_resolution_clock::now();
			frame_since_last_update = 0;

			return true;
		}
		return false;
	}

	float GetFPS() const noexcept { return fps; }
};

