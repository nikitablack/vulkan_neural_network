#pragma once

#include <cstdint>

namespace common {

template <typename T>
class LCG {
public:
    explicit LCG(uint32_t seed) : m_state{seed} {}

public:
    auto next() noexcept -> T {
        m_state = (m_a * m_state + m_c) % m_m;
        auto const v{static_cast<T>(m_state) / static_cast<T>(m_m)};

        return v * static_cast<T>(2) - static_cast<T>(1);
    }

private:
    static uint32_t constexpr m_a{1664525};
    static uint32_t constexpr m_c{1013904223};
    static uint32_t constexpr m_m{0xFFFFFFFF};
    uint32_t m_state;
};

}  // namespace common
