#pragma once

#include <string>
#include <vector>

namespace impl {
namespace graphics {

struct RequiredDeviceExtensions {
    static auto get() noexcept -> std::vector<std::string> const&;
    static auto print() noexcept -> void;
};

}  // namespace graphics
}  // namespace impl
