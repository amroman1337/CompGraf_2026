#include "application.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct GlobalUniforms {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec4 baseColor;
};

static_assert(sizeof(GlobalUniforms) == 208);
static_assert(offsetof(GlobalUniforms, baseColor) == 192);

const std::vector<Vertex> CUBE_VERTICES = {
    {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 1.0f}},
    {{ 0.5f,  0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}},
    {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f,  0.5f, -0.5f}, {1.0f, 1.0f, 0.0f}},
    {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}
};

const std::vector<uint32_t> CUBE_INDICES = {
    0, 1, 2,  2, 3, 0,
    5, 4, 7,  7, 6, 5,
    4, 0, 3,  3, 7, 4,
    1, 5, 6,  6, 2, 1,
    3, 2, 6,  6, 7, 3,
    4, 5, 1,  1, 0, 4
};

struct ObjectTransform {
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 rotation{ 0.0f, 0.0f, 0.0f };
    glm::vec3 scale{ 1.0f, 1.0f, 1.0f };
    float color[3] = { 1.0f, 1.0f, 1.0f };
};

struct AppState {
    bool isPerspective = true;
    bool isAnimating = false;
    float animSpeed = 0.5f;
    float animRadius = 2.0f;
    double animTime = 0.0;
    bool animationStarted = false;
    float animHeight = 0.6f;
    int verticalWaves = 3;
    float rotationSpeed = 45.0f;

    ObjectTransform cube1{ glm::vec3(-1.2f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f), {1.0f, 0.8f, 0.3f} };
    ObjectTransform cube2{ glm::vec3(1.2f, 0.0f, 0.0f), glm::vec3(0.0f, 45.0f, 0.0f), glm::vec3(0.8f), {0.2f, 0.8f, 1.0f} };
};

static AppState g_appState;

namespace application {

    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;

    VkBuffer global_uniform_buffer_1 = VK_NULL_HANDLE;
    VmaAllocation global_uniform_buffer_allocation_1 = VK_NULL_HANDLE;
    GlobalUniforms* global_uniform_buffer_memory_1 = nullptr;

    VkBuffer global_uniform_buffer_2 = VK_NULL_HANDLE;
    VmaAllocation global_uniform_buffer_allocation_2 = VK_NULL_HANDLE;
    GlobalUniforms* global_uniform_buffer_memory_2 = nullptr;

    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    VkDescriptorSet descriptor_set_1 = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_2 = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    void check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": VkResult=" +
                                     std::to_string(static_cast<int>(result)));
    }

std::vector<uint32_t> readShader(const char* name, const char* legacyName) {
    namespace fs = std::filesystem;
    std::vector<fs::path> roots;
    auto addParents = [&](fs::path root) {
        for (int i = 0; i < 4 && !root.empty(); ++i) {
            roots.push_back(root);
            const auto parent = root.parent_path();
            if (parent == root) break;
            root = parent;
        }
    };
#ifdef LAB_SHADER_DIR
    roots.emplace_back(LAB_SHADER_DIR);
#else
    roots.emplace_back(fs::current_path());
#endif
    for (const char* filename : {name, legacyName}) {
        for (const auto& root : roots) {
            for (const char* sub : {"shaders", "", "build-debug/shaders",
                                    "build-debug", "shaders/compiled"}) {
                const auto path = root / sub / filename;
                std::ifstream file(path, std::ios::binary | std::ios::ate);
                if (!file) continue;
                const auto end = file.tellg();
                if (end < std::streamoff(20) || end % std::streamoff(4) != 0)
                    continue;
                std::vector<uint32_t> words(static_cast<size_t>(end) / 4);
                file.seekg(0);
                file.read(reinterpret_cast<char*>(words.data()),
                          static_cast<std::streamsize>(end));
                if (file && words[0] == 0x07230203u) {
                    std::cout << "Loaded shader: " << path.string() << '\n';
                    return words;
                }
            }
        }
    }
    throw std::runtime_error(std::string("Cannot find ") + name +
        ". Put the supplied basic.*.spv in the project shaders/ directory "
        "and run from the project root.");
}
    VkShaderModule loadShaderModule(const char* name, const char* legacyName) {
        try {
            const auto code = readShader(name, legacyName);
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = code.size() * sizeof(uint32_t);
            info.pCode = code.data();
            VkShaderModule module = VK_NULL_HANDLE;
            check(vkCreateShaderModule(graphics::internal::context.device,
                                      &info, nullptr, &module), "vkCreateShaderModule");
            return module;
        } catch (const std::exception& e) {
            std::cerr << "Shader error: " << e.what() << std::endl;
            return VK_NULL_HANDLE;
        }
    }

    bool initialize() {
        auto& context = graphics::internal::context;
        try {

            const VkBufferCreateInfo vertex_buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(Vertex) * CUBE_VERTICES.size(),
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            const VmaAllocationCreateInfo vertex_alloc_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            Vertex* vertex_mapped_memory = nullptr;
            check(vmaCreateBuffer(context.allocator, &vertex_buffer_info, &vertex_alloc_info,
                &vertex_buffer, &vertex_buffer_allocation, nullptr), "vmaCreateBuffer");
            check(vmaMapMemory(context.allocator, vertex_buffer_allocation, (void**)&vertex_mapped_memory), "vmaMapMemory");

            auto coloredVertices = CUBE_VERTICES;
            for (auto& vertex : coloredVertices)
                vertex.color = vertex.position + glm::vec3(0.5f);
            memcpy(vertex_mapped_memory, coloredVertices.data(), vertex_buffer_info.size);
            const VkResult vertexFlush = vmaFlushAllocation(context.allocator,
                vertex_buffer_allocation, 0, vertex_buffer_info.size);
            vmaUnmapMemory(context.allocator, vertex_buffer_allocation);
            check(vertexFlush, "flush vertices");

            const VkBufferCreateInfo index_buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(uint32_t) * CUBE_INDICES.size(),
                .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            uint32_t* index_mapped_memory = nullptr;
            check(vmaCreateBuffer(context.allocator, &index_buffer_info, &vertex_alloc_info,
                &index_buffer, &index_buffer_allocation, nullptr), "vmaCreateBuffer");
            check(vmaMapMemory(context.allocator, index_buffer_allocation, (void**)&index_mapped_memory), "vmaMapMemory");
            memcpy(index_mapped_memory, CUBE_INDICES.data(), index_buffer_info.size);
            const VkResult indexFlush = vmaFlushAllocation(context.allocator,
                index_buffer_allocation, 0, index_buffer_info.size);
            vmaUnmapMemory(context.allocator, index_buffer_allocation);
            check(indexFlush, "flush indices");

            const VkBufferCreateInfo uniform_buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = (sizeof(GlobalUniforms) + 0xf) & ~0xf,
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            check(vmaCreateBuffer(context.allocator, &uniform_buffer_info, &vertex_alloc_info,
                &global_uniform_buffer_1, &global_uniform_buffer_allocation_1, nullptr), "vmaCreateBuffer");
            check(vmaMapMemory(context.allocator, global_uniform_buffer_allocation_1, (void**)&global_uniform_buffer_memory_1), "vmaMapMemory");

            check(vmaCreateBuffer(context.allocator, &uniform_buffer_info, &vertex_alloc_info,
                &global_uniform_buffer_2, &global_uniform_buffer_allocation_2, nullptr), "vmaCreateBuffer");
            check(vmaMapMemory(context.allocator, global_uniform_buffer_allocation_2, (void**)&global_uniform_buffer_memory_2), "vmaMapMemory");

            const VkDescriptorSetLayoutBinding descriptor_set_bindings[] = {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                }
            };
            const VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = descriptor_set_bindings,
            };
            check(vkCreateDescriptorSetLayout(context.device, &descriptor_layout_info, nullptr, &descriptor_set_layout), "vkCreateDescriptorSetLayout");

            const VkDescriptorPoolSize pool_sizes[] = {
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 }
            };
            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2,
                .poolSizeCount = 1,
                .pPoolSizes = pool_sizes,
            };
            check(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");

            VkDescriptorSetLayout layouts[2] = { descriptor_set_layout, descriptor_set_layout };
            const VkDescriptorSetAllocateInfo alloc_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 2,
                .pSetLayouts = layouts,
            };

            VkDescriptorSet descriptor_sets[2];
            check(vkAllocateDescriptorSets(context.device, &alloc_info, descriptor_sets), "vkAllocateDescriptorSets");
            descriptor_set_1 = descriptor_sets[0];
            descriptor_set_2 = descriptor_sets[1];

            const VkDescriptorBufferInfo ubo_descriptor_info_1 = {
                .buffer = global_uniform_buffer_1,
                .offset = 0,
                .range = sizeof(GlobalUniforms),
            };
            const VkDescriptorBufferInfo ubo_descriptor_info_2 = {
                .buffer = global_uniform_buffer_2,
                .offset = 0,
                .range = sizeof(GlobalUniforms),
            };

            const VkWriteDescriptorSet descriptor_writes[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_1,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &ubo_descriptor_info_1,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_2,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &ubo_descriptor_info_2,
                }
            };
            vkUpdateDescriptorSets(context.device, 2, descriptor_writes, 0, nullptr);

            // конвейер
            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &descriptor_set_layout,
            };
            check(vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

            VkShaderModule vert_shader = loadShaderModule("basic.vert.spv", "vert.spv");
            VkShaderModule frag_shader = loadShaderModule("basic.frag.spv", "frag.spv");

            if (!vert_shader || !frag_shader) {
                if (vert_shader) vkDestroyShaderModule(context.device, vert_shader, nullptr);
                if (frag_shader) vkDestroyShaderModule(context.device, frag_shader, nullptr);
                throw std::runtime_error("Cannot load compatible cube shaders");
            }
            std::cerr << "Creating cube pipeline..." << std::endl;
            const VkPipelineShaderStageCreateInfo shader_stages[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_shader,
                    .pName = "main",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = frag_shader,
                    .pName = "main",
                }
            };

            const VkVertexInputBindingDescription vertex_bindings[] = {
                {.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX }
            };
            const VkVertexInputAttributeDescription vertex_attributes[] = {
                {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
                {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color) }
            };
            const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = vertex_bindings,
                .vertexAttributeDescriptionCount = 2,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };
            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            };
            const VkPipelineRasterizationStateCreateInfo rasterizer = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .lineWidth = 1.0f,
            };
            const VkPipelineMultisampleStateCreateInfo multisampling = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };
            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };
            const VkPipelineColorBlendAttachmentState color_blend_attachment = {
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };
            const VkPipelineColorBlendStateCreateInfo color_blending = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &color_blend_attachment,
            };
            const VkDynamicState dynamic_states_array[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states_array,
            };

            const VkGraphicsPipelineCreateInfo pipeline_info = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = shader_stages,
                .pVertexInputState = &vertex_input_info,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blending,
                .pDynamicState = &dynamic_state,
                .layout = pipeline_layout,
                .renderPass = context.render_pass,
                .subpass = 0,
            };

            const VkResult pipelineResult = vkCreateGraphicsPipelines(
                context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

            vkDestroyShaderModule(context.device, frag_shader, nullptr);
            vkDestroyShaderModule(context.device, vert_shader, nullptr);
            check(pipelineResult, "vkCreateGraphicsPipelines");
            std::cerr << "Cube pipeline ready" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Application initialization failed: " << e.what() << std::endl;
            shutdown();
            return false;
        }
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        if (!context.device) return;
        vkDeviceWaitIdle(context.device);
        if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
        if (descriptor_set_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
        if (global_uniform_buffer_memory_1)
            vmaUnmapMemory(context.allocator, global_uniform_buffer_allocation_1);
        if (global_uniform_buffer_memory_2)
            vmaUnmapMemory(context.allocator, global_uniform_buffer_allocation_2);
        if (global_uniform_buffer_1)
            vmaDestroyBuffer(context.allocator, global_uniform_buffer_1, global_uniform_buffer_allocation_1);
        if (global_uniform_buffer_2)
            vmaDestroyBuffer(context.allocator, global_uniform_buffer_2, global_uniform_buffer_allocation_2);
        if (index_buffer) vmaDestroyBuffer(context.allocator, index_buffer, index_buffer_allocation);
        if (vertex_buffer) vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
        pipeline = VK_NULL_HANDLE;
        pipeline_layout = VK_NULL_HANDLE;
        descriptor_pool = VK_NULL_HANDLE;
        descriptor_set_layout = VK_NULL_HANDLE;
        descriptor_set_1 = descriptor_set_2 = VK_NULL_HANDLE;
        global_uniform_buffer_1 = global_uniform_buffer_2 = VK_NULL_HANDLE;
        vertex_buffer = index_buffer = VK_NULL_HANDLE;
        global_uniform_buffer_memory_1 = global_uniform_buffer_memory_2 = nullptr;
        global_uniform_buffer_allocation_1 = global_uniform_buffer_allocation_2 = VK_NULL_HANDLE;
        vertex_buffer_allocation = index_buffer_allocation = VK_NULL_HANDLE;
    }

    void update(double time) {
        ImGui::Begin("Lab 1");

        ImGui::Text("Projection Type (Extra 1):");
        if (ImGui::RadioButton("Perspective", g_appState.isPerspective))
            g_appState.isPerspective = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !g_appState.isPerspective))
            g_appState.isPerspective = false;
        ImGui::Separator();

        if (ImGui::CollapsingHeader("Cube 1 Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Pos 1 (XYZ)", &g_appState.cube1.position.x, 0.05f);
            ImGui::DragFloat3("Rot 1 (deg)", &g_appState.cube1.rotation.x, 1.0f);
            ImGui::DragFloat3("Scale 1", &g_appState.cube1.scale.x, 0.05f, 0.1f, 5.0f);
            ImGui::ColorEdit3("Color 1", g_appState.cube1.color);
        }

        if (ImGui::CollapsingHeader("Cube 2 Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Pos 2 (XYZ)", &g_appState.cube2.position.x, 0.05f);
            ImGui::DragFloat3("Rot 2 (deg)", &g_appState.cube2.rotation.x, 1.0f);
            ImGui::DragFloat3("Scale 2", &g_appState.cube2.scale.x, 0.05f, 0.1f, 5.0f);
            ImGui::ColorEdit3("Color 2", g_appState.cube2.color);
        }

        ImGui::Separator();
        ImGui::Text("Animation:");
        if (ImGui::Button(g_appState.isAnimating ? "Pause" : "Start Animation")) {
            g_appState.isAnimating = !g_appState.isAnimating;
            if (g_appState.isAnimating) g_appState.animationStarted = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset animation")) {
            g_appState.isAnimating = false;
            g_appState.animationStarted = false;
            g_appState.animTime = 0.0;
        }
        ImGui::SliderFloat("Height", &g_appState.animHeight, 0.0f, 2.0f);
        ImGui::SliderInt("Vertical waves", &g_appState.verticalWaves, 1, 6);
        ImGui::SliderFloat("Rotation speed", &g_appState.rotationSpeed, -180.0f, 180.0f);
        ImGui::SliderFloat("Speed", &g_appState.animSpeed, 0.01f, 5.0f);
        ImGui::SliderFloat("Radius", &g_appState.animRadius, 0.5f, 5.0f);

        ImGui::End();
        static double lastTime = time;
        const double trueDeltaTime = std::clamp(time - lastTime, 0.0, 0.1);
        lastTime = time;

        if (g_appState.isAnimating) {
            g_appState.animTime += trueDeltaTime * g_appState.animSpeed;
        }

    }


    void updateGpuUniforms() {
        glm::mat4 view = glm::lookAt(
            glm::vec3(5.0f, 4.0f, 10.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        auto& context = graphics::internal::context;
        float width = static_cast<float>(context.swapchain_extent.width);
        float height = static_cast<float>(context.swapchain_extent.height);
        float aspect = (height > 0.0f) ? (width / height) : 1.0f;

        glm::mat4 proj;
        if (g_appState.isPerspective) {
            proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        }
        else {
            float orthoSize = 3.5f;
            proj = glm::ortho(-orthoSize * aspect, orthoSize * aspect, -orthoSize, orthoSize, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        glm::mat4 model1 = glm::mat4(1.0f);
        glm::vec3 pos1 = g_appState.cube1.position;
        if (g_appState.animationStarted) {
            const double t = g_appState.animTime;
            pos1.x += float(g_appState.animRadius * std::sin(t));
            pos1.y += float(g_appState.animHeight * std::sin(g_appState.verticalWaves * t));
            pos1.z += float(0.5 * g_appState.animRadius * std::sin(2.0 * t));
        }
        model1 = glm::translate(model1, pos1);
        model1 = glm::rotate(model1, glm::radians(g_appState.cube1.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        float extraRot1 = float(std::fmod(g_appState.animTime * g_appState.rotationSpeed, 360.0));
        model1 = glm::rotate(model1, glm::radians(g_appState.cube1.rotation.y + extraRot1), glm::vec3(0.0f, 1.0f, 0.0f));
        model1 = glm::rotate(model1, glm::radians(g_appState.cube1.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model1 = glm::scale(model1, g_appState.cube1.scale);

        GlobalUniforms ubo1{};
        ubo1.model = model1;
        ubo1.view = view;
        ubo1.projection = proj;
        ubo1.baseColor = glm::vec4(g_appState.cube1.color[0], g_appState.cube1.color[1], g_appState.cube1.color[2], 1.0f);

        if (global_uniform_buffer_memory_1) {
            memcpy(global_uniform_buffer_memory_1, &ubo1, sizeof(ubo1));
            check(vmaFlushAllocation(context.allocator, global_uniform_buffer_allocation_1,
                                    0, sizeof(ubo1)), "flush cube 1 uniforms");
        }

        glm::mat4 model2 = glm::mat4(1.0f);
        glm::vec3 pos2 = g_appState.cube2.position;
        if (g_appState.animationStarted) {
            const double t = g_appState.animTime;
            pos2.x -= float(g_appState.animRadius * std::sin(t));
            pos2.y -= float(g_appState.animHeight * std::sin(g_appState.verticalWaves * t));
            pos2.z -= float(0.5 * g_appState.animRadius * std::sin(2.0 * t));
        }
        model2 = glm::translate(model2, pos2);
        model2 = glm::rotate(model2, glm::radians(g_appState.cube2.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        float extraRot2 = float(std::fmod(-g_appState.animTime * g_appState.rotationSpeed, 360.0));
        model2 = glm::rotate(model2, glm::radians(g_appState.cube2.rotation.y + extraRot2), glm::vec3(0.0f, 1.0f, 0.0f));
        model2 = glm::rotate(model2, glm::radians(g_appState.cube2.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model2 = glm::scale(model2, g_appState.cube2.scale);

        GlobalUniforms ubo2{};
        ubo2.model = model2;
        ubo2.view = view;
        ubo2.projection = proj;
        ubo2.baseColor = glm::vec4(g_appState.cube2.color[0], g_appState.cube2.color[1], g_appState.cube2.color[2], 1.0f);

        if (global_uniform_buffer_memory_2) {
            memcpy(global_uniform_buffer_memory_2, &ubo2, sizeof(ubo2));
            check(vmaFlushAllocation(context.allocator, global_uniform_buffer_allocation_2,
                                    0, sizeof(ubo2)), "flush cube 2 uniforms");
        }
    }

    void render(const graphics::internal::FrameData& fd) {
        if (pipeline == VK_NULL_HANDLE || !fd.command_buffer) return;

        auto& context = graphics::internal::context;
        updateGpuUniforms();

        const VkViewport viewport = {
            .x = 0, .y = 0,
            .width = static_cast<float>(context.swapchain_extent.width),
            .height = static_cast<float>(context.swapchain_extent.height),
            .minDepth = 0, .maxDepth = 1,
        };
        const VkRect2D scissor = { .extent = context.swapchain_extent };

        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_set_1, 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(CUBE_INDICES.size()), 1, 0, 0, 0);

        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_set_2, 0, nullptr);
        vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(CUBE_INDICES.size()), 1, 0, 0, 0);

    }

} // namespace application