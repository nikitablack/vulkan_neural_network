#pragma once

#include <cstdint>

namespace impl {

auto swap_endian(uint32_t val) noexcept -> uint32_t;

}  // namespace impl
