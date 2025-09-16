#include <fmt/ranges.h>

#include <common/LCG.hpp>
#include <cstdlib>
#include <impl/NeuralNetwork.hpp>
#include <impl/graphics/GraphicsManager.hpp>

auto main(int /* argc */, char* /* argv */[]) -> int {
    common::LCG<float> lcg{42};

    impl::graphics::GraphicsManager graphicsManager{};
    graphicsManager.init();

    std::vector<float> input(784);
    std::vector<float> out(10);

    impl::NeuralNetwork nn{graphicsManager, {input.size(), 100, out.size()}, lcg};

    nn.forward(graphicsManager, input, &out);

    nn.clear();

    fmt::println("{}", out);

    // impl::graphics::DeviceBuffer weights{};
    // impl::graphics::DeviceBuffer biases{};
    // impl::graphics::DeviceBuffer previousLayerValues{};
    // impl::graphics::DeviceBuffer values{};
    // impl::graphics::HostVisibleBuffer valuesHost{};

    // {
    //     uint32_t constexpr N{4};
    //     uint32_t constexpr S{N * sizeof(float)};

    //     std::vector<float> vs{1.0f, 2.0f, 3.0f, 4.0f};

    //     weights.init(graphicsManager.allocator,  //
    //                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
    //                  S);

    //     impl::graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
    //                                             weights,  //
    //                                             reinterpret_cast<uint8_t*>(vs.data()),  //
    //                                             S,
    //                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
    //                                             VK_ACCESS_SHADER_READ_BIT,  //
    //                                             graphicsManager.computeQueue);
    // }

    // {
    //     uint32_t constexpr N{2};
    //     uint32_t constexpr S{N * sizeof(float)};

    //     std::vector<uint8_t> data(S);
    //     std::vector<float> vs{1.0f, 2.0f};
    //     std::memcpy(data.data(), vs.data(), S);

    //     biases.init(graphicsManager.allocator,  //
    //                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
    //                 S);

    //     impl::graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
    //                                             biases,  //
    //                                             data,  //
    //                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
    //                                             VK_ACCESS_SHADER_READ_BIT,  //
    //                                             graphicsManager.computeQueue);
    // }

    // {
    //     uint32_t constexpr N{2};
    //     uint32_t constexpr S{N * sizeof(float)};

    //     std::vector<uint8_t> data(S);
    //     std::vector<float> vs{5.0f, 6.0f};
    //     std::memcpy(data.data(), vs.data(), S);

    //     previousLayerValues.init(graphicsManager.allocator,  //
    //                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
    //                              S);

    //     impl::graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
    //                                             previousLayerValues,  //
    //                                             data,  //
    //                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
    //                                             VK_ACCESS_SHADER_READ_BIT,  //
    //                                             graphicsManager.computeQueue);
    // }

    // {
    //     uint32_t constexpr N{2};
    //     uint32_t constexpr S{N * sizeof(float)};

    //     values.init(graphicsManager.allocator,  //
    //                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
    //                 S);

    //     valuesHost.init(graphicsManager.allocator,  //
    //                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
    //                     S,  //
    //                     true);
    // }

    // graphicsManager.forward(weights, biases, previousLayerValues, values);

    // // read
    // {
    //     auto const commandBuffer{graphicsManager.commandManager.getCommandBufferBegin()};

    //     impl::graphics::utils::set_buffer_barrier(commandBuffer,  //
    //                                               values,  //
    //                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
    //                                               VK_ACCESS_SHADER_WRITE_BIT,  //
    //                                               VK_PIPELINE_STAGE_TRANSFER_BIT,  //
    //                                               VK_ACCESS_TRANSFER_READ_BIT);

    //     VkBufferCopy region{};
    //     region.srcOffset = 0;
    //     region.dstOffset = 0;
    //     region.size = values.getSize();

    //     vkCmdCopyBuffer(commandBuffer, values.getBuffer(), valuesHost.getBuffer(), 1, &region);

    //     if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    //         fmt::println("Failed to end copy command buffer.");
    //     }

    //     VkSubmitInfo submitInfo{};
    //     submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    //     submitInfo.pNext = nullptr;
    //     submitInfo.waitSemaphoreCount = 0;
    //     submitInfo.pWaitSemaphores = nullptr;
    //     submitInfo.pWaitDstStageMask = nullptr;
    //     submitInfo.commandBufferCount = 1;
    //     submitInfo.pCommandBuffers = &commandBuffer;
    //     submitInfo.signalSemaphoreCount = 0;
    //     submitInfo.pSignalSemaphores = nullptr;

    //     if (vkQueueSubmit(graphicsManager.computeQueue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
    //         fmt::println("Failed to submit staging command buffer.");
    //     }

    //     if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
    //         fmt::println("Failed to wait staging queue.");
    //     }

    //     auto const* const data{static_cast<float const*>(valuesHost.getMappedData())};
    //     fmt::println("Value: {} {}", data[0], data[1]);
    // }

    // weights.destroy();
    // biases.destroy();
    // previousLayerValues.destroy();
    // values.destroy();
    // valuesHost.destroy();

    graphicsManager.clear();

    return EXIT_SUCCESS;
}