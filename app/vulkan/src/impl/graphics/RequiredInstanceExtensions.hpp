#pragma once

#include <string>
#include <vector>

namespace impl {
namespace graphics {

struct RequiredInstanceExtensions {
    static auto get() noexcept -> const std::vector<std::string>&;
    static auto print() noexcept -> void;
};

}  // namespace graphics
}  // namespace impl
