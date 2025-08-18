#pragma once

#include <string>
#include <vector>

namespace impl {

auto load_labels(std::string const& name) noexcept -> std::vector<uint8_t>;

}  // namespace impl
