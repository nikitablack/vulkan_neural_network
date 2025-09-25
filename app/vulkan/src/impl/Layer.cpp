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

        graphics::utils::init_buffer_sync(graphicsManager,  //
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
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                    neuronCount * sizeof(float));

        graphicsManager.debugUtils.setName(biases.getBuffer(), "Layer biases.");

        graphics::utils::init_buffer_sync(graphicsManager,  //
                                          biases,  //
                                          reinterpret_cast<uint8_t*>(hdata.data()),  //
                                          hdata.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_READ_BIT,  //
                                          graphicsManager.computeQueue);

        delta.init(graphicsManager.allocator,  //
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  //
                   neuronCount * sizeof(float));

        graphicsManager.debugUtils.setName(delta.getBuffer(), "Layer delta.");
    }

    values.init(graphicsManager.allocator,  //
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
                neuronCount * sizeof(float));

    graphicsManager.debugUtils.setName(values.getBuffer(), "Layer values.");

    m_activateDescriptorSet = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                                graphicsManager.descriptorPool,  //
                                                                graphicsManager.descriptorSetLayout);
    graphicsManager.debugUtils.setName(m_activateDescriptorSet, "Activate descriptor set.");

    m_inferDescriptorSet = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                             graphicsManager.descriptorPool,  //
                                                             graphicsManager.descriptorSetLayout);
    graphicsManager.debugUtils.setName(m_inferDescriptorSet, "Infer descriptor set.");

    m_updateDescriptorSet = graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                              graphicsManager.descriptorPool,  //
                                                              graphicsManager.descriptorSetLayout);
    graphicsManager.debugUtils.setName(m_updateDescriptorSet, "Update descriptor set.");
}

auto Layer::activate(graphics::GraphicsManager const& graphicsManager,  //
                     VkCommandBuffer commandBuffer,  //
                     Layer const& prevLayer,  //
                     graphics::DeviceBuffer const& batchIndex,  //
                     bool infer  //
                     ) -> void {
    if (inputSize() != prevLayer.size()) {
        throw std::runtime_error{"Mismatch between layers sizes."};
    }

    // the very first layer - input - comes in batches
    // in this case, for the first hidden layer, we should offset the input
    // for all other layers we should not add an offset
    // see forward.comp
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

    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};

    if (infer) {
        descriptorSet = m_inferDescriptorSet;

        // Lazy update: it's enough to update the set once, all followed calls will use the same descriptor set
        if (!m_inferDescriptorSetUpdated) {
            m_inferDescriptorSetUpdated = true;

            // it's enough to update the set once, all followed calls will use the same descriptor set
            graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                                   m_inferDescriptorSet,  //
                                                   weights,  //
                                                   biases,  //
                                                   prevLayer.values,  //
                                                   values,  //
                                                   batchIndex);
        }
    } else {
        descriptorSet = m_activateDescriptorSet;

        // Lazy update: it's enough to update the set once, all followed calls will use the same descriptor set
        if (!m_activateDescriptorSetUpdated) {
            m_activateDescriptorSetUpdated = true;

            graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                                   m_activateDescriptorSet,  //
                                                   weights,  //
                                                   biases,  //
                                                   prevLayer.values,  //
                                                   values,  //
                                                   batchIndex);
        }
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &descriptorSet,  //
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

    // if previous layer - input layer - no need in barrier, since its values never change
    if (prevLayer.inputSize() != 0) {
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
                   graphics::DeviceBuffer const& batchIndex  //
                   ) -> void {
    if (delta.getSize() != size() * sizeof(float)) {
        throw std::runtime_error{"Mismatch between neuron count and delta count."};
    }

    // the very first layer - input - comes in batches
    // in this case, for the first hidden layer, we should offset the input
    // for all other layers we should not add an offset
    // see forward.comp
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

    // Lazy update: it's enough to update the set once, all followed calls will use the same descriptor set
    if (!m_updateDescriptorSetUpdated) {
        m_updateDescriptorSetUpdated = true;

        graphics::utils::update_descriptor_set(graphicsManager.device,  //
                                               m_updateDescriptorSet,  //
                                               weights,  //
                                               biases,  //
                                               prevLayer.values,  //
                                               delta,  //
                                               batchIndex);
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &m_updateDescriptorSet,  //
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

auto Layer::inputSize() const noexcept -> size_t {
    return m_inputSize;
}

auto Layer::clear() noexcept -> void {
    weights.destroy();
    biases.destroy();
    values.destroy();
    delta.destroy();
}

}  // namespace impl
