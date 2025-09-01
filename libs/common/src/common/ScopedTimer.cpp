#include <fmt/format.h>

#include <common/ScopedTimer.hpp>

namespace common {

ScopedTimer::ScopedTimer(std::string&& msg) noexcept : m_timer{}, m_msg{std::move(msg)} {
    m_timer.start();
}

ScopedTimer::~ScopedTimer() noexcept {
    fmt::println("{}{} ms", m_msg, m_timer.stop());
}

}  // namespace common
