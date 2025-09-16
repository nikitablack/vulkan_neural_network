#include <impl/graphics/features/Features.hpp>

namespace impl {
namespace graphics {
namespace features {

auto Features::print() noexcept -> void {}

auto Features::check(VkPhysicalDevice /* physicalDevice */) noexcept -> bool {
    bool result{true};

    return result;
}

auto Features::get_chain() noexcept -> VkPhysicalDeviceFeatures2* {
    return nullptr;
}

}  // namespace features
}  // namespace graphics
}  // namespace impl
