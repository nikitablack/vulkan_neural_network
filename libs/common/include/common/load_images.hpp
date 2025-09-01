#pragma once

#include <common/Float.hpp>
#include <string>
#include <vector>

namespace common {

auto load_images(std::string const& name) noexcept -> std::vector<std::vector<Float>>;

}  // namespace common
