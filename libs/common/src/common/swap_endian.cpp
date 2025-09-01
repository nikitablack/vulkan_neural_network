#include <common/swap_endian.hpp>

namespace common {

auto swap_endian(uint32_t val) noexcept -> uint32_t {
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) | ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
}

}  // namespace common
