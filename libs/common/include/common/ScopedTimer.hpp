#pragma once

#include <chrono>
#include <common/Timer.hpp>
#include <string>

namespace common {

class ScopedTimer {
public:
    ScopedTimer(std::string&& msg) noexcept;
    ~ScopedTimer() noexcept;

private:
    Timer m_timer;
    std::string m_msg;
};

}  // namespace common
