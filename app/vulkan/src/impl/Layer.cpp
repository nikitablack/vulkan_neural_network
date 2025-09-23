#include <fmt/core.h>

#include <common/LCG.hpp>
#include <impl/Layer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/get_push_constant_data.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <impl/graphics/utils/update_descriptor_set.hpp>
#include <stdexcept>
#include <type_traits>

namespace impl {

Layer::Layer(graphics::GraphicsManager& graphicsManager,  //
             size_t neuronCount,  //
             size_t inputCount,  //
             common::LCG<float>& lcg  //
             )
    : m_size{neuronCount}, m_inputSize{inputCount} {
    if (inputCount > 0) {
        std::vector<float> hdata{};

        // initialize weights
        hdata.reserve(neuronCount * inputCount);

        for (size_t r{0}; r < neuronCount; ++r) {
            for (size_t c{0}; c < inputCount; ++c) {
                hdata.push_back(lcg.next());
            }
        }

        weights.init(graphicsManager.allocator,  //
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                     neuronCount * inputCount * sizeof(float));

        graphicsManager.debugUtils.setName(weights.getBuffer(), "Layer weights.");

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          weights,  //
                                          reinterpret_cast<uint8_t*>(hdata.data()),  //
                                          hdata.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_READ_BIT,  //
                                          graphicsManager.computeQueue);

        // initialize biases
        hdata.clear();
        hdata.reserve(neuronCount);

        for (size_t r{0}; r < neuronCount; ++r) {
            hdata.push_back(lcg.next());
        }

        biases.init(graphicsManager.allocator,  //
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                    neuronCount * sizeof(float));

        graphicsManager.debugUtils.setName(biases.getBuffer(), "Layer biases.");

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          biases,  //
                                          reinterpret_cast<uint8_t*>(hdata.data()),  //
                                          hdata.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_READ_BIT,  //
                                          graphicsManager.computeQueue);

        delta.init(graphicsManager.allocator,  //
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                   neuronCount * sizeof(float));

        graphicsManager.debugUtils.setName(delta.getBuffer(), "Layer delta.");
    }

    values.init(graphicsManager.allocator,  //
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                neuronCount * sizeof(float));

    graphicsManager.debugUtils.setName(values.getBuffer(), "Layer values.");

    for (size_t i{0}; i < m_activateDescriptorSets.size(); ++i) {
        m_activateDescriptorSets[i] = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                        graphicsManager.descriptorPool,  //
                                                                        graphicsManager.descriptorSetLayout);
        graphicsManager.debugUtils.setName(m_activateDescriptorSets[i], fmt::format("Activate descriptor set {}.", i));
    }

    for (size_t i{0}; i < m_updateDescriptorSets.size(); ++i) {
        m_updateDescriptorSets[i] = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                      graphicsManager.descriptorPool,  //
                                                                      graphicsManager.descriptorSetLayout);
        graphicsManager.debugUtils.setName(m_updateDescriptorSets[i], fmt::format("Update descriptor set {}.", i));
    }
}

auto Layer::activate(graphics::GraphicsManager const& graphicsManager,  //
                     VkCommandBuffer commandBuffer,  //
                     Layer const& prevLayer,  //
                     uint32_t iterationIndex,  //
                     graphics::DeviceBuffer const& batchIndex,  //
                     bool infer  //
                     ) -> void {
    if (inputSize() != prevLayer.size()) {
        throw std::runtime_error{"Mismatch between layers sizes."};
    }

    // the very first layer - input - comes in batches
    bool const batchedInput{prevLayer.inputSize() == 0};

    // see forward.comp
    auto const pushConstData{graphics::utils::get_push_constant_data(static_cast<uint32_t>(size()),  //
                                                                     static_cast<uint32_t>(inputSize()),  //
                                                                     batchedInput ? uint32_t{1} : uint32_t{0})};

    vkCmdPushConstants(commandBuffer,  //
                       graphicsManager.pipelineLayout,  //
                       VK_SHADER_STAGE_COMPUTE_BIT,  //
                       0,  //
                       static_cast<uint32_t>(pushConstData.size()),  //
                       pushConstData.data());

    auto const currDescriptorSet{m_activateDescriptorSets[iterationIndex]};

    if (!m_descSetToUpdated[currDescriptorSet] || infer) {
        m_descSetToUpdated[currDescriptorSet] = true;

        graphics::utils::BufferUpdateInfo const weightsBufferUpdateInfo{weights.getBuffer(),  //
                                                                        weights.getSize(),  //
                                                                        0};
        graphics::utils::BufferUpdateInfo const biasesBufferUpdateInfo{biases.getBuffer(),  //
                                                                       biases.getSize(),  //
                                                                       0};
        graphics::utils::BufferUpdateInfo const inputValuesBufferUpdateInfo{prevLayer.values.getBuffer(),  //
                                                                            prevLayer.values.getSize(),  //
                                                                            0};
        graphics::utils::BufferUpdateInfo const valuesBufferUpdateInfo{values.getBuffer(),  //
                                                                       values.getSize(),  //
                                                                       0};
        graphics::utils::BufferUpdateInfo const batchIndexBufferUpdateInfo{batchIndex.getBuffer(),  //
                                                                           batchIndex.getSize(),  //
                                                                           0};

        graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                               currDescriptorSet,  //
                                               weightsBufferUpdateInfo,  //
                                               biasesBufferUpdateInfo,  //
                                               inputValuesBufferUpdateInfo,  //
                                               valuesBufferUpdateInfo,  //
                                               batchIndexBufferUpdateInfo);
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &currDescriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.forwardPipeline);

    graphics::utils::set_buffer_barrier(commandBuffer,  //
                                        weights,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_WRITE_BIT,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    graphics::utils::set_buffer_barrier(commandBuffer,  //
                                        biases,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_WRITE_BIT,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // set barrier for a previous layer
    if (prevLayer.values.getBuffer() != VK_NULL_HANDLE) {
        graphics::utils::set_buffer_barrier(commandBuffer,  //
                                            prevLayer.values,  //
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                            VK_ACCESS_SHADER_WRITE_BIT,  //
                                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                            VK_ACCESS_SHADER_READ_BIT);
    }

    uint32_t constexpr GROUP_SIZE{64};  // see forward.comp
    uint32_t const groupCount{(static_cast<uint32_t>(size()) + GROUP_SIZE - 1) / GROUP_SIZE};
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
}

auto Layer::update(graphics::GraphicsManager const& graphicsManager,  //
                   VkCommandBuffer commandBuffer,  //
                   Layer const& prevLayer,  //
                   float learningRate,  //
                   uint32_t iterationIndex,  //
                   graphics::DeviceBuffer const& batchIndex  //
                   ) -> void {
    if (delta.getSize() != size() * sizeof(float)) {
        throw std::runtime_error{"Mismatch between neuron count and delta count."};
    }

    // the very first layer - input - comes in batches
    bool const batchedInput{prevLayer.inputSize() == 0};

    // see forward.comp
    auto const pushConstData{graphics::utils::get_push_constant_data(static_cast<uint32_t>(size()),  //
                                                                     static_cast<uint32_t>(inputSize()),  //
                                                                     learningRate,  //
                                                                     batchedInput ? uint32_t{1} : uint32_t{0})};

    vkCmdPushConstants(commandBuffer,  //
                       graphicsManager.pipelineLayout,  //
                       VK_SHADER_STAGE_COMPUTE_BIT,  //
                       0,  //
                       static_cast<uint32_t>(pushConstData.size()),  //
                       pushConstData.data());

    auto const currDescriptorSet{m_updateDescriptorSets[iterationIndex]};

    if (!m_descSetToUpdated[currDescriptorSet]) {
        m_descSetToUpdated[currDescriptorSet] = true;

        graphics::utils::BufferUpdateInfo const weightsBufferUpdateInfo{weights.getBuffer(),  //
                                                                        weights.getSize(),  //
                                                                        0};
        graphics::utils::BufferUpdateInfo const biasesBufferUpdateInfo{biases.getBuffer(),  //
                                                                       biases.getSize(),  //
                                                                       0};
        graphics::utils::BufferUpdateInfo const inputValuesBufferUpdateInfo{prevLayer.values.getBuffer(),  //
                                                                            prevLayer.values.getSize(),  //
                                                                            0};
        graphics::utils::BufferUpdateInfo const deltaBufferUpdateInfo{delta.getBuffer(),  //
                                                                      delta.getSize(),  //
                                                                      0};
        graphics::utils::BufferUpdateInfo const batchIndexBufferUpdateInfo{batchIndex.getBuffer(),  //
                                                                           batchIndex.getSize(),  //
                                                                           0};

        graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                               currDescriptorSet,  //
                                               weightsBufferUpdateInfo,  //
                                               biasesBufferUpdateInfo,  //
                                               inputValuesBufferUpdateInfo,  //
                                               deltaBufferUpdateInfo,  //
                                               batchIndexBufferUpdateInfo);
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &currDescriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.updatePipeline);

    graphics::utils::set_buffer_barrier(commandBuffer,  //
                                        delta,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_WRITE_BIT,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_READ_BIT);

    uint32_t constexpr GROUP_SIZE{16};  // see update.comp
    uint32_t const groupCountX{(static_cast<uint32_t>(inputSize()) + GROUP_SIZE - 1) / GROUP_SIZE};
    uint32_t const groupCountY{(static_cast<uint32_t>(size()) + GROUP_SIZE - 1) / GROUP_SIZE};
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}

auto Layer::size() const noexcept -> size_t {
    return m_size;
}

auto Layer::sizeBytes() const noexcept -> size_t {
    return m_size * sizeof(float);
}

auto Layer::inputSize() const noexcept -> size_t {
    return m_inputSize;
}

auto Layer::inputSizeBytes() const noexcept -> size_t {
    return m_inputSize * sizeof(float);
}

auto Layer::clear() noexcept -> void {
    weights.destroy();
    biases.destroy();
    values.destroy();
    delta.destroy();
}

}  // namespace impl
