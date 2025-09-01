#pragma once

#include <chrono>

namespace common {

class Timer {
public:
    Timer() noexcept;

public:
    auto start() noexcept -> void;
    auto stop() noexcept -> double;

private:
    decltype(std::chrono::high_resolution_clock::now()) m_startTime;
};

}  // namespace common
