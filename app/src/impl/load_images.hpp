#pragma once

#include <impl/Float.hpp>
#include <string>
#include <vector>

namespace impl {

auto load_images(std::string const& name) noexcept -> std::vector<std::vector<Float>>;

}  // namespace impl
