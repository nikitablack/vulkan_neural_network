#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/allocate_descriptor_set.hpp>
#include <impl/graphics/calculate_output_delta.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/get_push_constant_data.hpp>
#include <impl/graphics/utils/update_descriptor_set.hpp>
#include <unordered_map>

namespace {

std::unordered_map<VkDescriptorSet, bool> m_descSetToUpdated{};

}

namespace impl {
namespace graphics {

auto calculate_output_delta(GraphicsManager& graphicsManager,  //
                            VkCommandBuffer commandBuffer,  //
                            VkDescriptorSet descriptorSet,  //
                            uint32_t neuronCount,  //
                            DeviceBuffer const& values,  //
                            DeviceBuffer const& expectedOutput,  //
                            DeviceBuffer const& delta,  //
                            DeviceBuffer const& batchIndex  //
                            ) -> void {
    // see delta.comp
    auto const pushConstData{utils::get_push_constant_data(neuronCount,  //
                                                           uint32_t{0},  // not used for for this dispatch
                                                           uint32_t{1})};

    vkCmdPushConstants(commandBuffer,  //
                       graphicsManager.pipelineLayout,  //
                       VK_SHADER_STAGE_COMPUTE_BIT,  //
                       0,  //
                       static_cast<uint32_t>(pushConstData.size()),  //
                       pushConstData.data());

    if (!m_descSetToUpdated[descriptorSet]) {
        m_descSetToUpdated[descriptorSet] = true;

        graphics::utils::BufferUpdateInfo const valuesBufferUpdateInfo{values.getBuffer(),  //
                                                                       values.getSize(),  //
                                                                       0};
        graphics::utils::BufferUpdateInfo const expectedOutputBufferUpdateInfo{expectedOutput.getBuffer(),  //
                                                                               expectedOutput.getSize(),  //
                                                                               0};
        graphics::utils::BufferUpdateInfo const deltaBufferUpdateInfo{delta.getBuffer(),  //
                                                                      delta.getSize(),  //
                                                                      0};
        graphics::utils::BufferUpdateInfo const batchIndexBufferUpdateInfo{batchIndex.getBuffer(),  //
                                                                           batchIndex.getSize(),  //
                                                                           0};

        // see delta.comp
        utils::update_descriptor_set(graphicsManager.device,  //
                                     descriptorSet,  //
                                     valuesBufferUpdateInfo,  // not used for for this dispatch
                                     valuesBufferUpdateInfo,  //
                                     expectedOutputBufferUpdateInfo,  //
                                     deltaBufferUpdateInfo,  //
                                     batchIndexBufferUpdateInfo);
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &descriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.deltaPipeline);

    // set barrier for a previous layer
    utils::set_buffer_barrier(commandBuffer,  //
                              values,  //
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                              VK_ACCESS_SHADER_WRITE_BIT,  //
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                              VK_ACCESS_SHADER_READ_BIT);

    uint32_t constexpr GROUP_SIZE{32};  // see delta.comp
    uint32_t const groupCount{(neuronCount + GROUP_SIZE - 1) / GROUP_SIZE};
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
}

}  // namespace graphics
}  // namespace impl
