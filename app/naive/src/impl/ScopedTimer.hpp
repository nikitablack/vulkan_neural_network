#pragma once

#include <chrono>
#include <impl/Timer.hpp>
#include <string>

namespace impl {

class ScopedTimer {
public:
    ScopedTimer(std::string&& msg) noexcept;
    ~ScopedTimer() noexcept;

private:
    Timer m_timer;
    std::string m_msg;
};

}  // namespace impl
