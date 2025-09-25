#include <impl/Layer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/get_command_buffer.hpp>
#include <impl/graphics/set_input_and_target_data.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <stdexcept>

namespace {

auto set_input_data(impl::graphics::GraphicsManager& graphicsManager,  //
                    VkCommandBuffer commandBuffer,  //
                    impl::Layer& inputLayer,  //
                    std::vector<std::vector<float>> const& input  //
                    ) -> impl::graphics::HostVisibleBuffer {
    auto const singleInputSizeBytes{input[0].size() * sizeof(float)};
    auto const totalInputSizeBytes{input.size() * singleInputSizeBytes};

    if (inputLayer.values.getSize() < totalInputSizeBytes) {
        if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to wait queue on train."};
        }

        inputLayer.values.destroy();

        inputLayer.values.init(graphicsManager.allocator,  //
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                               totalInputSizeBytes);

        graphicsManager.debugUtils.setName(inputLayer.values.getBuffer(), "Layer values.");
    }

    impl::graphics::HostVisibleBuffer stagingBuffer{graphicsManager.allocator,  //
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                                                    totalInputSizeBytes};

    // since the input is a vector of vectors, we cant' just memcpy input directly
    // instead, we need to iterate and memcpy each entry individually
    for (size_t inputIdx{0}; inputIdx < input.size(); ++inputIdx) {
        auto const& v{input[inputIdx]};

        stagingBuffer.copyData(reinterpret_cast<uint8_t const*>(v.data()),  //
                               singleInputSizeBytes,  //
                               singleInputSizeBytes * inputIdx);
    }

    impl::graphics::utils::init_buffer(commandBuffer,  //
                                       inputLayer.values,  //
                                       stagingBuffer,  //
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                       VK_ACCESS_SHADER_WRITE_BIT);

    return stagingBuffer;
}

auto set_expected_output_data(impl::graphics::GraphicsManager& graphicsManager,  //
                              VkCommandBuffer commandBuffer,  //
                              impl::Layer& outputLayer,  //
                              impl::graphics::DeviceBuffer& expectedOutput,  //
                              std::vector<uint8_t> const& target  //
                              ) -> impl::graphics::HostVisibleBuffer {
    auto const singleOutputSizeBytes{outputLayer.size() * sizeof(float)};
    auto const totalOutputSizeBytes{target.size() * singleOutputSizeBytes};

    if (expectedOutput.getSize() < totalOutputSizeBytes) {
        if (vkQueueWaitIdle(graphicsManager.computeQueue.queue) != VK_SUCCESS) {
            throw std::runtime_error{"Failed to wait queue on train."};
        }

        expectedOutput.destroy();

        expectedOutput.init(graphicsManager.allocator,  //
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                            totalOutputSizeBytes);

        graphicsManager.debugUtils.setName(expectedOutput.getBuffer(), "Expected output.");
    }

    impl::graphics::HostVisibleBuffer stagingBuffer{graphicsManager.allocator,  //
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                                                    totalOutputSizeBytes};

    std::vector<float> tmp(outputLayer.size(), 0.0);

    // Since the expected data size is equal to the output layer size, but the target is just vector of integers, we
    // need to expand it.
    //
    // For example, the target[0] = 5 means that for the corresponding input 0, the result should be 5.
    //
    // But the output layer expects not a single number, but an array filled with zero exccept an element N, where N is
    // our target. So for our example and an output layer with size 10, the expected data should be
    // [0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f], i.e. the array of size 10 with 5-th element equal
    // to 1.
    for (size_t outputIdx{0}; outputIdx < target.size(); ++outputIdx) {
        auto const t{target[outputIdx]};

        // set the correct position, for each input there's a corresponding output
        tmp[t] = 1.0f;

        stagingBuffer.copyData(reinterpret_cast<uint8_t const*>(tmp.data()),  //
                               singleOutputSizeBytes,  //
                               singleOutputSizeBytes * outputIdx);

        // reset the position after copy
        tmp[t] = 0.0f;
    }

    impl::graphics::utils::init_buffer(commandBuffer,  //
                                       expectedOutput,  //
                                       stagingBuffer,  //
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                       VK_ACCESS_SHADER_WRITE_BIT);

    return stagingBuffer;
}

}  // namespace

namespace impl {
namespace graphics {

auto set_input_and_target_data(GraphicsManager& graphicsManager,  //
                               Layer& inputLayer,  //
                               std::vector<std::vector<float>> const& input,  //
                               Layer& outputLayer,  //
                               DeviceBuffer& expectedOutput,  //
                               std::vector<uint8_t> const& target  //
                               ) -> void {
    auto const commandBuffer{get_command_buffer_begin(graphicsManager.device, graphicsManager.commandPool)};
    std::unordered_map<VkCommandBuffer, std::vector<HostVisibleBuffer>> initData{};

    auto const inputStagingBuffer{set_input_data(graphicsManager, commandBuffer, inputLayer, input)};
    initData[commandBuffer].push_back(inputStagingBuffer);

    // init expected output all at once
    auto const targetStagingBuffer{set_expected_output_data(graphicsManager,  //
                                                            commandBuffer,  //
                                                            outputLayer,  //
                                                            expectedOutput,  //
                                                            target)};
    initData[commandBuffer].push_back(targetStagingBuffer);

    utils::submit_init_data_sync(std::move(initData), graphicsManager.computeQueue);
}

}  // namespace graphics
}  // namespace impl