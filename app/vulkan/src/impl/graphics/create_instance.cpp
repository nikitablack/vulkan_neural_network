#include <impl/graphics/RequiredInstanceExtensions.hpp>
#include <impl/graphics/create_instance.hpp>
#include <stdexcept>

namespace impl {
namespace graphics {

auto create_instance() -> VkInstance {
    const auto& requiredExtensions{RequiredInstanceExtensions::get()};

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = nullptr;
    appInfo.pApplicationName = "Game";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = nullptr;
    appInfo.engineVersion = 0;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<char const*> requiredExtensionsStr{};
    requiredExtensionsStr.reserve(requiredExtensions.size());

    for (auto const& extension : requiredExtensions) {
        requiredExtensionsStr.push_back(extension.data());
    }

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.pApplicationInfo = &appInfo;
    info.enabledLayerCount = 0;
    info.ppEnabledLayerNames = nullptr;
    info.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    info.ppEnabledExtensionNames = requiredExtensionsStr.data();

    VkInstance instance;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create instance."};
    }

    return instance;
}

}  // namespace graphics
}  // namespace impl
