#include <vulkan/vulkan.h>

#include <impl/graphics/RequiredInstanceExtensions.hpp>
#include <impl/graphics/check_required_instance_extensions.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto check_required_instance_extensions() -> void {
    RequiredInstanceExtensions::print();

    uint32_t extensionCount;

    if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to enumerate instance extension properties."};
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);

    if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to enumerate instance extension properties."};
    }

    for (auto const& reqExt : RequiredInstanceExtensions::get()) {
        for (auto const& avExt : availableExtensions) {
            if (reqExt == avExt.extensionName) {
                goto cnt;
            }
        }

        throw std::runtime_error{"Instance extension \"" + reqExt + "\" is not supported."};

    cnt:;
    }
}

}  // namespace graphics
}  // namespace impl
