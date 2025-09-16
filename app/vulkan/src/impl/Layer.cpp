#include <fmt/core.h>

#include <common/LCG.hpp>
#include <impl/Layer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/get_push_constant_data.hpp>
#include <impl/graphics/utils/init_helper.hpp>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename T, typename... Ts>
constexpr auto make_array(T&& first, Ts&&... rest) {
    using U = std::decay_t<T>;
    static_assert((std::is_same_v<U, std::decay_t<Ts>> && ...), "All arguments must have the same type");

    return std::array<U, 1 + sizeof...(Ts)>{{std::forward<T>(first), std::forward<Ts>(rest)...}};
}

template <typename... Args>
auto update_descriptor_set(VkDevice device, VkDescriptorSet descriptorSet, Args&&... buffers) -> void {
    auto const arr{make_array(std::forward<Args>(buffers)...)};

    std::array<VkWriteDescriptorSet, arr.size()> writeDescriptorSets{};
    std::array<VkDescriptorBufferInfo, arr.size()> infos{};

    for (size_t i{0}; i < arr.size(); ++i) {
        infos[i].buffer = arr[i].getBuffer();
        infos[i].offset = 0;
        infos[i].range = arr[i].getSize();

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

}  // namespace

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
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,  //
                    neuronCount * sizeof(float));

        graphics::utils::init_buffer_sync(graphicsManager.commandManager,  //
                                          biases,  //
                                          reinterpret_cast<uint8_t*>(hdata.data()),  //
                                          hdata.size() * sizeof(float),  //
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                          VK_ACCESS_SHADER_READ_BIT,  //
                                          graphicsManager.computeQueue);
    }

    values.init(
        graphicsManager.allocator,  //
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  //
        neuronCount * sizeof(float));
}

auto Layer::activate(graphics::GraphicsManager const& graphicsManager,  //
                     VkCommandBuffer commandBuffer,  //
                     Layer const& prevLayer  //
                     ) -> void {
    if (inputSize() != prevLayer.size()) {
        throw std::runtime_error{"Mismatch between values size and weights size."};
    }

    auto const descriptorSet{graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                               graphicsManager.descriptorPool,  //
                                                               graphicsManager.descriptorSetLayout)};
    graphicsManager.debugUtils.setName(descriptorSet, "Forward descriptor set.");

    // see forward.comp
    auto const pushConstData{graphics::utils::get_push_constant_data(static_cast<uint32_t>(size()),  //
                                                                     static_cast<uint32_t>(inputSize()))};

    vkCmdPushConstants(commandBuffer,  //
                       graphicsManager.pipelineLayout,  //
                       VK_SHADER_STAGE_COMPUTE_BIT,  //
                       0,  //
                       static_cast<uint32_t>(pushConstData.size()),  //
                       pushConstData.data());

    update_descriptor_set(graphicsManager.device,  //
                          descriptorSet,  //
                          weights,  //
                          biases,  //
                          prevLayer.values,  //
                          values);

    // std::array<graphics::DeviceBuffer, 4> buffers{weights,  //
    //                                               biases,  //
    //                                               prevLayer.values,  //
    //                                               values};

    // std::array<VkWriteDescriptorSet, buffers.size()> writeDescriptorSets{};
    // std::array<VkDescriptorBufferInfo, buffers.size()> infos{};

    // for (size_t i{0}; i < buffers.size(); ++i) {
    //     infos[i].buffer = buffers[i].getBuffer();
    //     infos[i].offset = 0;
    //     infos[i].range = buffers[i].getSize();

    //     writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    //     writeDescriptorSets[i].pNext = nullptr;
    //     writeDescriptorSets[i].dstSet = descriptorSet;
    //     writeDescriptorSets[i].dstBinding = static_cast<uint32_t>(i);
    //     writeDescriptorSets[i].dstArrayElement = 0;
    //     writeDescriptorSets[i].descriptorCount = 1;
    //     writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //     writeDescriptorSets[i].pImageInfo = nullptr;
    //     writeDescriptorSets[i].pBufferInfo = &infos[i];
    //     writeDescriptorSets[i].pTexelBufferView = nullptr;
    // }

    // vkUpdateDescriptorSets(graphicsManager.device,  //
    //                        writeDescriptorSets.size(),  //
    //                        writeDescriptorSets.data(),  //
    //                        0,  //
    //                        nullptr);

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &descriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.forwardPipeline);

    // set barrier for a previous layer
    graphics::utils::set_buffer_barrier(commandBuffer,  //
                                        prevLayer.values,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_WRITE_BIT,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_READ_BIT);

    uint32_t constexpr GROUP_SIZE{64};  // see forward.comp
    uint32_t const groupCount{(static_cast<uint32_t>(size()) + GROUP_SIZE - 1) / GROUP_SIZE};
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
}

auto Layer::update(graphics::GraphicsManager const& graphicsManager,  //
                   VkCommandBuffer commandBuffer,  //
                   Layer const& prevLayer,  //
                   float learningRate,  //
                   graphics::DeviceBuffer const& delta  //
                   ) -> void {
    if (delta.getSize() != size() * sizeof(float)) {
        throw std::runtime_error{"Mismatch between neuron count and delta count."};
    }

    auto const descriptorSet{graphics::allocate_descriptor_set(graphicsManager.device,  //
                                                               graphicsManager.descriptorPool,  //
                                                               graphicsManager.descriptorSetLayout)};
    graphicsManager.debugUtils.setName(descriptorSet, "Update descriptor set.");

    // see forward.comp
    auto const pushConstData{graphics::utils::get_push_constant_data(static_cast<uint32_t>(size()),  //
                                                                     static_cast<uint32_t>(inputSize()),  //
                                                                     learningRate)};

    vkCmdPushConstants(commandBuffer,  //
                       graphicsManager.pipelineLayout,  //
                       VK_SHADER_STAGE_COMPUTE_BIT,  //
                       0,  //
                       static_cast<uint32_t>(pushConstData.size()),  //
                       pushConstData.data());

    update_descriptor_set(graphicsManager.device,  //
                          descriptorSet,  //
                          weights,  //
                          biases,  //
                          prevLayer.values,  //
                          delta);

    // std::array<graphics::DeviceBuffer, 4> buffers{weights,  //
    //                                               biases,  //
    //                                               prevLayer.values,  //
    //                                               delta};

    // std::array<VkWriteDescriptorSet, buffers.size()> writeDescriptorSets{};
    // std::array<VkDescriptorBufferInfo, buffers.size()> infos{};

    // for (size_t i{0}; i < buffers.size(); ++i) {
    //     infos[i].buffer = buffers[i].getBuffer();
    //     infos[i].offset = 0;
    //     infos[i].range = buffers[i].getSize();

    //     writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    //     writeDescriptorSets[i].pNext = nullptr;
    //     writeDescriptorSets[i].dstSet = descriptorSet;
    //     writeDescriptorSets[i].dstBinding = static_cast<uint32_t>(i);
    //     writeDescriptorSets[i].dstArrayElement = 0;
    //     writeDescriptorSets[i].descriptorCount = 1;
    //     writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //     writeDescriptorSets[i].pImageInfo = nullptr;
    //     writeDescriptorSets[i].pBufferInfo = &infos[i];
    //     writeDescriptorSets[i].pTexelBufferView = nullptr;
    // }

    // vkUpdateDescriptorSets(graphicsManager.device,  //
    //                        writeDescriptorSets.size(),  //
    //                        writeDescriptorSets.data(),  //
    //                        0,  //
    //                        nullptr);

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &descriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.updatePipeline);

    // set barrier for a previous layer
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

    graphics::utils::set_buffer_barrier(commandBuffer,  //
                                        prevLayer.values,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_WRITE_BIT,  //
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                                        VK_ACCESS_SHADER_READ_BIT);

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
}

}  // namespace impl
