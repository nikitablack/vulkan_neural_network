#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/update_current_batch_index.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/update_descriptor_set.hpp>
#include <unordered_map>

namespace {

std::unordered_map<VkDescriptorSet, bool> m_descSetToUpdated{};

}

namespace impl {
namespace graphics {

auto update_current_batch_index(GraphicsManager& graphicsManager,  //
                                VkCommandBuffer commandBuffer,  //
                                VkDescriptorSet descriptorSet,  //
                                DeviceBuffer const& currBatchIndex,  //
                                DeviceBuffer const& batchIndices  //
                                ) -> void {
    if (!m_descSetToUpdated[descriptorSet]) {
        m_descSetToUpdated[descriptorSet] = true;

        graphics::utils::BufferUpdateInfo const currBatchIndexBufferUpdateInfo{currBatchIndex.getBuffer(),  //
                                                                               currBatchIndex.getSize(),  //
                                                                               0};
        graphics::utils::BufferUpdateInfo const batchIndicesBufferUpdateInfo{batchIndices.getBuffer(),  //
                                                                             batchIndices.getSize(),  //
                                                                             0};

        // see batch_index.comp
        utils::update_descriptor_set(graphicsManager.device,  //
                                     descriptorSet,  //
                                     currBatchIndexBufferUpdateInfo,  //
                                     batchIndicesBufferUpdateInfo,  //
                                     currBatchIndexBufferUpdateInfo,  // not used
                                     currBatchIndexBufferUpdateInfo  // not used
        );
    }

    vkCmdBindDescriptorSets(commandBuffer,  //
                            VK_PIPELINE_BIND_POINT_COMPUTE,  //
                            graphicsManager.pipelineLayout,  //
                            0,  //
                            1,  //
                            &descriptorSet,  //
                            0,  //
                            nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsManager.batchIndexPipeline);

    // set barrier for all next layer
    utils::set_buffer_barrier(commandBuffer,  //
                              currBatchIndex,  //
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                              VK_ACCESS_SHADER_WRITE_BIT,  //
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  //
                              VK_ACCESS_SHADER_READ_BIT);

    // see batch_index.comp
    vkCmdDispatch(commandBuffer, 1, 1, 1);
}

}  // namespace graphics
}  // namespace impl
