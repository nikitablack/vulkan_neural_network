#include <fmt/core.h>

#include <cmrc/cmrc.hpp>
#include <impl/graphics/create_shader_module.hpp>
#include <stdexcept>
#include <vector>

CMRC_DECLARE(shaders);

namespace impl {
namespace graphics {

auto create_shader_module(VkDevice device, std::string const& name) -> VkShaderModule {
    auto const fs{cmrc::shaders::get_filesystem()};
    auto const shader{fs.open(name)};

    if (shader.size() == 0 || (shader.size() % 4 != 0)) {
        throw std::runtime_error{fmt::format("Shader file {} size should be a nonzero multiple of 4.", name)};
    }

    std::vector<uint32_t> spirv(shader.size() / 4);
    memcpy(spirv.data(), shader.begin(), shader.size());

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.codeSize = shader.size();
    info.pCode = spirv.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &info, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error{fmt::format("failed to create shader module for the shader {}.", name)};
    }

    return shaderModule;
}

}  // namespace graphics
}  // namespace impl
