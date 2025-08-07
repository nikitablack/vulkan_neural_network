#pragma once

#include <string>
#include <vector>

namespace impl {

auto load_images(std::string const& name) noexcept -> std::vector<std::vector<double>>;

}  // namespace impl
