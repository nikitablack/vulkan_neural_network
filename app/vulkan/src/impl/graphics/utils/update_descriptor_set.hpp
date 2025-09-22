#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <utility>

namespace impl {
namespace graphics {
namespace utils {

struct BufferUpdateInfo {
    VkBuffer buffer;
    VkDeviceSize range;
    VkDeviceSize offset;
};

template <typename T, typename... Ts>
constexpr auto make_array(T&& first, Ts&&... rest) noexcept {
    using U = std::decay_t<T>;
    static_assert((std::is_same_v<U, std::decay_t<Ts>> && ...), "All arguments must have the same type");

    return std::array<U, 1 + sizeof...(Ts)>{{first, rest...}};
}

template <typename... Args>
auto update_descriptor_set(VkDevice device, VkDescriptorSet descriptorSet, Args&&... buffers) noexcept -> void {
    auto const arr{make_array(std::forward<Args>(buffers)...)};

    std::array<VkWriteDescriptorSet, arr.size()> writeDescriptorSets{};
    std::array<VkDescriptorBufferInfo, arr.size()> infos{};

    for (size_t i{0}; i < arr.size(); ++i) {
        infos[i].buffer = arr[i].buffer;
        infos[i].offset = arr[i].offset;
        infos[i].range = arr[i].range;

        writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[i].pNext = nullptr;
        writeDescriptorSets[i].dstSet = descriptorSet;
        writeDescriptorSets[i].dstBinding = static_cast<uint32_t>(i);
        writeDescriptorSets[i].dstArrayElement = 0;
        writeDescriptorSets[i].descriptorCount = 1;
        writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[i].pImageInfo = nullptr;
        writeDescriptorSets[i].pBufferInfo = &infos[i];
        writeDescriptorSets[i].pTexelBufferView = nullptr;
    }

    vkUpdateDescriptorSets(device,  //
                           writeDescriptorSets.size(),  //
                           writeDescriptorSets.data(),  //
                           0,  //
                           nullptr);
}

}  // namespace utils
}  // namespace graphics
}  // namespace impl
