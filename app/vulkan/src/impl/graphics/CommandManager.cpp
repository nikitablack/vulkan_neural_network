#include <cassert>
#include <impl/graphics/CommandManager.hpp>
#include <stdexcept>

namespace {

constexpr uint32_t NUM_BUFFERS_TO_ALLOCATE_AT_ONCE{10};

}

namespace impl {
namespace graphics {

auto CommandManager::init(VkDevice device, uint32_t queueFamily) -> void {
    m_device = device;

    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = queueFamily;

    if (vkCreateCommandPool(m_device, &info, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to create command pool."};
    }

    m_availableCommandBuffers.reserve(NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);
    m_busyCommandBuffers.reserve(NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);
}

auto CommandManager::getCommandBuffer() -> VkCommandBuffer {
    if (!m_device) {
        throw std::runtime_error{"Attempt to get a command buffer from not initialized instance of CommandManager."};
    }

    // if there are no available command buffers, allocate NUM_BUFFERS_TO_ALLOCATE_AT_ONCE and keep in a vector
    if (m_availableCommandBuffers.empty()) {
        VkCommandBufferAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.pNext = nullptr;
        info.commandPool = m_commandPool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = NUM_BUFFERS_TO_ALLOCATE_AT_ONCE;

        m_availableCommandBuffers.resize(NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);
        if (vkAllocateCommandBuffers(m_device, &info, m_availableCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to allocate command buffers."};
        }
    }

    auto const commandBuffer{m_availableCommandBuffers.back()};
    m_availableCommandBuffers.pop_back();

    // take a buffer from the list of available and add it to the busy vector
    m_busyCommandBuffers.push_back(commandBuffer);

    return commandBuffer;
}

auto CommandManager::getCommandBufferBegin() -> VkCommandBuffer {
    auto const commandBuffer{getCommandBuffer()};

    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &info) != VK_SUCCESS) {
        throw std::runtime_error{"Failed to begin command buffer."};
    }

    return commandBuffer;
}

auto CommandManager::tick() -> void {
    if (!m_device) {
        throw std::runtime_error{"Attempt to tick a not initialized instance of CommandManager."};
    }

    // rotate the vector of busy command buffers,
    // i.e. the 0th element becomes 1st, the 1st becomes second, and so on until CONCURRENT_FRAME_COUNT
    // the last element (CONCURRENT_FRAME_COUNT - 1) should reset all of its command buffers

    for (auto cb : m_busyCommandBuffers) {
        if (vkResetCommandBuffer(cb, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)) {
            throw std::runtime_error{"Failed to reset command buffer."};
        }
    }

    m_availableCommandBuffers.insert(m_availableCommandBuffers.end(),  //
                                     m_busyCommandBuffers.begin(),  //
                                     m_busyCommandBuffers.end());
    m_busyCommandBuffers.clear();
}

auto CommandManager::clear() noexcept -> void {
    if (!m_device) {
        return;
    }

    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;

    m_device = VK_NULL_HANDLE;

    m_availableCommandBuffers.clear();
    m_busyCommandBuffers.clear();
}

}  // namespace graphics
}  // namespace impl
