#pragma once

#include <cstdint>

namespace impl {
namespace graphics {

struct RequiredApiVersion {
    static uint32_t constexpr MAJOR{1};
    static uint32_t constexpr MINOR{0};
};

}  // namespace graphics
}  // namespace impl
