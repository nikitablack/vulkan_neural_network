#include <impl/Timer.hpp>

namespace impl {

Timer::Timer() noexcept : m_startTime{std::chrono::high_resolution_clock::now()} {}

auto Timer::start() noexcept -> void {
    m_startTime = std::chrono::high_resolution_clock::now();
}

auto Timer::stop() noexcept -> double {
    auto const now{std::chrono::high_resolution_clock::now()};
    auto const durationNs{std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_startTime)};
    auto const durationMs{static_cast<double>(durationNs.count()) / 1'000'000.0};

    return durationMs;
}

}  // namespace impl
