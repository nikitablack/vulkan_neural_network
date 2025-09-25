#include <impl/graphics/DeviceBuffer.hpp>
#include <impl/graphics/GraphicsManager.hpp>
#include <impl/graphics/update_current_batch_index.hpp>
#include <impl/graphics/utils/barrier_helper.hpp>
#include <impl/graphics/utils/update_descriptor_set.hpp>
#include <unordered_map>

namespace impl {
namespace graphics {

auto update_current_batch_index(GraphicsManager& graphicsManager,  //
                                VkCommandBuffer commandBuffer,  //
                                VkDescriptorSet descriptorSet,  //
                                DeviceBuffer const& currBatchIndex  //
                                ) -> void {
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
