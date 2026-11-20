#include "application.hpp"

#include <imgui.h>
#include <png.h>
#include <functional>
#include <limits>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct alignas(16) SceneUniforms {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 ambientColorIntensity;
    glm::vec4 directionalDirection;
    glm::vec4 directionalColorIntensity;
    glm::uvec4 lightCounts;
};

struct alignas(16) ModelUniforms {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 albedo;
    glm::vec4 specularColor;
    glm::vec4 materialParameters;
};

struct alignas(16) GpuPointLight {
    glm::vec4 position;
    glm::vec4 colorIntensity;
};

struct alignas(16) GpuSpotLight {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 colorIntensity;
    glm::vec4 coneAngles;
};

static_assert(sizeof(SceneUniforms) == 144);
static_assert(offsetof(SceneUniforms, cameraPosition) == 64);
static_assert(offsetof(SceneUniforms, lightCounts) == 128);
static_assert(sizeof(ModelUniforms) == 176);
static_assert(offsetof(ModelUniforms, normalMatrix) == 64);
static_assert(offsetof(ModelUniforms, albedo) == 128);
static_assert(offsetof(ModelUniforms, materialParameters) == 160);
static_assert(sizeof(GpuPointLight) == 32);
static_assert(sizeof(GpuSpotLight) == 64);
static_assert(offsetof(GpuSpotLight, coneAngles) == 48);

struct ObjectTransform {
    glm::vec3 position{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 albedo{1.0f, 0.8f, 0.3f};
    glm::vec3 specularColor{1.0f};
    float shininess = 48.0f;
    int textureIndex = 0;
};

struct PointLight {
    glm::vec3 position{-1.5f, 1.5f, 2.0f};
    glm::vec3 color{1.0f, 0.65f, 0.3f};
    float intensity = 8.0f;
};

struct SpotLight {
    glm::vec3 position{0.0f, 3.0f, 3.0f};
    glm::vec3 direction{0.0f, -1.0f, -1.0f};
    glm::vec3 color{0.4f, 0.65f, 1.0f};
    float intensity = 18.0f;
    float innerAngle = 15.0f;
    float outerAngle = 25.0f;
    bool attachedToCamera = false;
};

struct Camera {
    glm::vec3 position{4.0f, 3.0f, 7.0f};
    float yaw = -120.0f;
    float pitch = -20.0f;
    float fov = 55.0f;
    float movementSpeed = 3.0f;
};

struct AppState {
    bool basicMode = false;
    bool effectEnabled = true;
    bool effectPlaying = true;
    double effectTime = 0.0;
    float effectSpeed = 1.0f;
    float distortion = 0.05f;
    float sampleBlend = 0.4f;
    float uvScale = 2.0f;
    glm::vec3 effectTint{1.0f};
    float emission = 0.0f;
    Camera camera;
    bool isPerspective = true;
    float orthoSize = 4.0f;
    glm::vec3 ambientColor{1.0f};
    float ambientIntensity = 0.08f;
    glm::vec3 directionalDirection{-0.5f, -1.0f, -0.4f};
    glm::vec3 directionalColor{1.0f, 0.95f, 0.85f};
    float directionalIntensity = 0.8f;
    bool directionalEnabled = true;

    bool isAnimating = false;
    bool animationStarted = false;
    double animTime = 0.0;
    float animSpeed = 0.5f;
    float animRadius = 2.0f;
    float animHeight = 0.6f;
    int verticalWaves = 3;
    float rotationSpeed = 45.0f;
    bool showFloor = true;

    ObjectTransform cube1;
    ObjectTransform cube2;
    ObjectTransform floor;
};

static AppState g_appState;

namespace application {

    struct Buffer {
        VkBuffer handle = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };

    struct Texture {
        std::string name;
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
    };
    struct alignas(16) EffectPush {
        glm::vec4 animation;
        glm::vec4 modulation;
        glm::uvec4 control;
    };
    static_assert(sizeof(EffectPush) == 48);
    std::vector<Texture> textures;
    VkSampler texture_sampler = VK_NULL_HANDLE;
    VkCommandPool upload_pool = VK_NULL_HANDLE;
    VkPipeline effect_pipeline = VK_NULL_HANDLE;
    VkDeviceSize model_stride = 0;
    Buffer vertex_buffer;
    Buffer index_buffer;
    Buffer scene_uniform_buffer;
    Buffer model_uniform_buffer;
    Buffer point_light_buffer;
    Buffer spot_light_buffer;
    uint32_t cube_index_count = 0;

    VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;
    VkDeviceSize max_storage_range = 0;
    uint32_t max_point_count = 0;
    uint32_t max_spot_count = 0;
    double previous_time = -1.0;

    void check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": VkResult=" +
                                     std::to_string(static_cast<int>(result)));
    }

    glm::vec3 safeNormalize(glm::vec3 value, glm::vec3 fallback) {
        const float lengthSquared = glm::dot(value, value);
        return lengthSquared > 1e-10f ? value / std::sqrt(lengthSquared) : fallback;
    }

    glm::vec3 cameraForward() {
        const float yaw = glm::radians(g_appState.camera.yaw);
        const float pitch = glm::radians(std::clamp(g_appState.camera.pitch, -89.0f, 89.0f));
        return safeNormalize(glm::vec3(std::cos(yaw) * std::cos(pitch),
                                       std::sin(pitch), std::sin(yaw) * std::cos(pitch)),
                             glm::vec3(0.0f, 0.0f, -1.0f));
    }

    void resetScene() {
        g_appState = AppState{};
        g_appState.cube1.position = {-1.2f, 0.0f, 0.0f};
        g_appState.cube2.position = {1.2f, 0.0f, 0.0f};
        g_appState.cube2.rotation = {0.0f, 45.0f, 0.0f};
        g_appState.cube2.scale = glm::vec3(0.8f);
        g_appState.cube2.albedo = {0.2f, 0.8f, 1.0f};
        g_appState.cube2.shininess = 80.0f;
        g_appState.floor.position = {0.0f, -0.8f, 0.0f};
        g_appState.floor.scale = {10.0f, 0.1f, 10.0f};
        g_appState.floor.albedo = glm::vec3(0.5f);
        g_appState.floor.specularColor = glm::vec3(0.15f);
        g_appState.floor.shininess = 24.0f;
        g_appState.cube1.albedo = glm::vec3(1.0f);
        g_appState.cube2.albedo = glm::vec3(1.0f);
        g_appState.floor.albedo = glm::vec3(1.0f);
        g_appState.cube2.textureIndex = 1;
        g_appState.floor.textureIndex = 3;
        point_lights.clear();
        spot_lights.clear();
    }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        auto& context = graphics::internal::context;
        if (size == 0) throw std::runtime_error("Cannot allocate an empty buffer");
        const VkBufferCreateInfo buffer_info = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo allocation_info = {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        Buffer result;
        VmaAllocationInfo info{};
        check(vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
                              &result.handle, &result.allocation, &info), "vmaCreateBuffer");
        result.mapped = info.pMappedData;
        result.size = size;
        if (!result.mapped) {
            vmaDestroyBuffer(context.allocator, result.handle, result.allocation);
            throw std::runtime_error("VMA returned an unmapped buffer");
        }
        return result;
    }

    void destroyBuffer(Buffer& buffer) {
        if (buffer.handle)
            vmaDestroyBuffer(graphics::internal::context.allocator,
                             buffer.handle, buffer.allocation);
        buffer = {};
    }

    void writeBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
        if (size == 0) return;
        if (!buffer.mapped || size > buffer.size)
            throw std::runtime_error("Invalid mapped buffer write");
        std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
        check(vmaFlushAllocation(graphics::internal::context.allocator,
                                buffer.allocation, 0, size), "vmaFlushAllocation");
    }

    void uploadCommands(const std::function<void(VkCommandBuffer)>& record) {
        auto& context = graphics::internal::context;
        const VkCommandBufferAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = upload_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(context.device, &alloc, &cmd), "upload command buffer");
        try {
            const VkCommandBufferBeginInfo begin = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            check(vkBeginCommandBuffer(cmd, &begin), "begin upload");
            record(cmd);
            check(vkEndCommandBuffer(cmd), "end upload");
            const VkSubmitInfo submit = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &cmd,
            };
            check(vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE), "submit upload");
            check(vkQueueWaitIdle(context.graphics_queue), "wait upload");
        } catch (...) {
            vkFreeCommandBuffers(context.device, upload_pool, 1, &cmd);
            throw;
        }
        vkFreeCommandBuffers(context.device, upload_pool, 1, &cmd);
    }

    Buffer createGpuBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
        auto& context = graphics::internal::context;
        Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer result;
        try {
            writeBuffer(staging, data, size);
            const VkBufferCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            VmaAllocationCreateInfo allocation{};
            allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            check(vmaCreateBuffer(context.allocator, &info, &allocation,
                                  &result.handle, &result.allocation, nullptr), "GPU buffer");
            result.size = size;
            uploadCommands([&](VkCommandBuffer cmd) {
                const VkBufferCopy copy = {0, 0, size};
                vkCmdCopyBuffer(cmd, staging.handle, result.handle, 1, &copy);
                const VkBufferMemoryBarrier barrier = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ?
                        VK_ACCESS_INDEX_READ_BIT : VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = result.handle, .offset = 0, .size = size,
                };
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
            });
        } catch (...) {
            destroyBuffer(staging);
            destroyBuffer(result);
            throw;
        }
        destroyBuffer(staging);
        return result;
    }

    std::filesystem::path texturePath(const char* filename) {
        namespace fs = std::filesystem;
        std::vector<fs::path> folders;
#ifdef LAB_TEXTURE_DIR
        folders.emplace_back(LAB_TEXTURE_DIR);
#endif
        folders.push_back(fs::current_path() / "textures");
#ifdef __linux__
        std::error_code error;
        auto exe = fs::read_symlink("/proc/self/exe", error);
        if (!error) folders.push_back(exe.parent_path().parent_path() / "textures");
#endif
        for (const auto& folder : folders)
            if (fs::is_regular_file(folder / filename)) return folder / filename;
        throw std::runtime_error(std::string("Missing texture: ") + filename);
    }

    void loadTextures() {
        auto& context = graphics::internal::context;
        const VkSamplerCreateInfo samplerInfo = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .minLod = 0.0f, .maxLod = 0.0f,
        };
        check(vkCreateSampler(context.device, &samplerInfo, nullptr, &texture_sampler), "sampler");
        const char* files[] = {"checker.png", "bricks.png", "stripes.png",
                               "vulcan.png", "rings.png", "mosaic.png"};
        textures.reserve(6);
        for (const char* filename : files) {
            png_image decoded{};
            decoded.version = PNG_IMAGE_VERSION;
            const std::string path = texturePath(filename).string();
            if (!png_image_begin_read_from_file(&decoded, path.c_str())) {
                const std::string message = decoded.message;
                png_image_free(&decoded);
                throw std::runtime_error("PNG read: " + path + ": " + message);
            }
            decoded.format = PNG_FORMAT_RGBA;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context.physical_device, &properties);
            if (!decoded.width || !decoded.height || decoded.width > properties.limits.maxImageDimension2D ||
                decoded.height > properties.limits.maxImageDimension2D) {
                png_image_free(&decoded);
                throw std::runtime_error("Texture dimensions exceed device limits");
            }
            const uint32_t width = decoded.width, height = decoded.height;
            std::vector<unsigned char> pixels;
            try { pixels.resize(PNG_IMAGE_SIZE(decoded)); }
            catch (...) { png_image_free(&decoded); throw; }
            if (!png_image_finish_read(&decoded, nullptr, pixels.data(), 0, nullptr)) {
                const std::string message = decoded.message;
                png_image_free(&decoded);
                throw std::runtime_error("PNG decode: " + message);
            }
            png_image_free(&decoded);
            textures.emplace_back();
            auto& texture = textures.back();
            texture.name = filename;
            Buffer staging = createBuffer(pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            try {
                writeBuffer(staging, pixels.data(), pixels.size());
                const VkImageCreateInfo imageInfo = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_SRGB,
                    .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
                    .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                };
                VmaAllocationCreateInfo allocation{};
                allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                allocation.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                check(vmaCreateImage(context.allocator, &imageInfo, &allocation,
                                     &texture.image, &texture.allocation, nullptr), "texture image");
                uploadCommands([&](VkCommandBuffer cmd) {
                    VkImageMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image = texture.image;
                    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {width, height, 1};
                    vkCmdCopyBufferToImage(cmd, staging.handle, texture.image,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                });
                const VkImageViewCreateInfo view = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = texture.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = VK_FORMAT_R8G8B8A8_SRGB,
                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                };
                check(vkCreateImageView(context.device, &view, nullptr, &texture.view), "texture view");
            } catch (...) { destroyBuffer(staging); throw; }
            destroyBuffer(staging);
        }
    }

    void createCube() {
        const std::array<glm::vec3, 8> positions = {{
            {-0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f},
            { 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
            {-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f},
            { 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}
        }};

        const std::array<std::array<uint32_t, 4>, 6> faces = {{
            {{0,1,2,3}}, {{5,4,7,6}}, {{4,0,3,7}},
            {{1,5,6,2}}, {{3,2,6,7}}, {{4,5,1,0}}
        }};
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        for (const auto& face : faces) {
            const glm::vec3 edge1 = positions[face[1]] - positions[face[0]];
            const glm::vec3 edge2 = positions[face[2]] - positions[face[0]];
            const glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));
            const uint32_t first = static_cast<uint32_t>(vertices.size());
            const std::array<glm::vec2, 4> uv = {{{0,1}, {1,1}, {1,0}, {0,0}}};
            for (size_t j = 0; j < face.size(); ++j)
                vertices.push_back({positions[face[j]], normal, uv[j]});
            for (uint32_t local : {0u, 1u, 2u, 2u, 3u, 0u})
                indices.push_back(first + local);
        }
        vertex_buffer = createGpuBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        index_buffer = createGpuBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cube_index_count = static_cast<uint32_t>(indices.size());
    }

    void updateDescriptor(VkDescriptorSet set, uint32_t binding,
                          VkDescriptorType type, const Buffer& buffer) {
        const VkDescriptorBufferInfo info = {
            .buffer = buffer.handle, .offset = 0, .range = buffer.size,
        };
        const VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = binding,
            .descriptorCount = 1,
            .descriptorType = type,
            .pBufferInfo = &info,
        };
        vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
    }

    void createDescriptorResources() {
        auto& context = graphics::internal::context;
        const VkDescriptorSetLayoutBinding scene_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3, .pBindings = scene_bindings,
        };
        check(vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
                                         &scene_set_layout), "scene layout");
        const VkDescriptorSetLayoutBinding model_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        const VkDescriptorSetLayoutCreateInfo model_layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2, .pBindings = model_bindings,
        };
        check(vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
                                         &model_set_layout), "material layout");
        const uint32_t count = static_cast<uint32_t>(textures.size());
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, count},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count},
        };
        const VkDescriptorPoolCreateInfo pool = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = count + 1, .poolSizeCount = 4, .pPoolSizes = sizes,
        };
        check(vkCreateDescriptorPool(context.device, &pool, nullptr, &descriptor_pool), "descriptor pool");
        std::vector<VkDescriptorSetLayout> layouts(count + 1, model_set_layout);
        layouts[0] = scene_set_layout;
        const VkDescriptorSetAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool, .descriptorSetCount = count + 1,
            .pSetLayouts = layouts.data(),
        };
        std::vector<VkDescriptorSet> sets(count + 1);
        check(vkAllocateDescriptorSets(context.device, &alloc, sets.data()), "descriptor sets");
        scene_descriptor_set = sets[0];
        scene_uniform_buffer = createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        updateDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_uniform_buffer);
        model_uniform_buffer = createBuffer(model_stride * 3, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        for (uint32_t i = 0; i < count; ++i) {
            textures[i].descriptor = sets[i + 1];
            const VkDescriptorBufferInfo buffer = {model_uniform_buffer.handle, 0, sizeof(ModelUniforms)};
            const VkDescriptorImageInfo image = {
                texture_sampler, textures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };
            const VkWriteDescriptorSet writes[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = textures[i].descriptor, .dstBinding = 0, .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = textures[i].descriptor, .dstBinding = 1, .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image,
                },
            };
            vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
        }
        point_light_buffer = createBuffer(sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        spot_light_buffer = createBuffer(sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const GpuPointLight emptyPoint{};
        const GpuSpotLight emptySpot{};
        writeBuffer(point_light_buffer, &emptyPoint, sizeof(emptyPoint));
        writeBuffer(spot_light_buffer, &emptySpot, sizeof(emptySpot));
        updateDescriptor(scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_light_buffer);
        updateDescriptor(scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_light_buffer);
    }

    void ensureLightCapacity(Buffer& buffer, size_t count, VkDeviceSize stride,
                             uint32_t binding) {
        const VkDeviceSize maxCount = max_storage_range / stride;
        const VkDeviceSize requiredCount = std::max<VkDeviceSize>(1, count);
        if (requiredCount > maxCount)
            throw std::runtime_error("Light storage exceeds the GPU's maxStorageBufferRange");
        if (buffer.size / stride >= requiredCount) return;
        VkDeviceSize capacity = std::max<VkDeviceSize>(1, buffer.size / stride);
        while (capacity < requiredCount) {
            if (capacity > maxCount / 2) { capacity = requiredCount; break; }
            capacity *= 2;
        }
        Buffer replacement = createBuffer(capacity * stride, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        updateDescriptor(scene_descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, replacement);
        destroyBuffer(buffer);
        buffer = replacement;
    }

    std::vector<uint32_t> readShader(const char* filename) {
        namespace fs = std::filesystem;
        std::vector<fs::path> folders;
#ifdef LAB_SHADER_DIR
        folders.emplace_back(LAB_SHADER_DIR);
#endif
        folders.push_back(fs::current_path() / "shaders");
        folders.push_back(fs::current_path() / "build-debug" / "shaders");
        folders.push_back(fs::current_path() / "build-release" / "shaders");
#ifdef __linux__
        std::error_code error;
        const auto exe = fs::read_symlink("/proc/self/exe", error);
        if (!error) folders.push_back(exe.parent_path() / "shaders");
#endif
        for (const auto& folder : folders) {
            const auto path = folder / filename;
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) continue;
            const auto size = static_cast<std::streamoff>(file.tellg());
            if (size < 20 || size % 4 != 0) continue;
            std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
            file.seekg(0);
            file.read(reinterpret_cast<char*>(words.data()), size);
            if (file && words[0] == 0x07230203u) {
                std::cerr << "Loaded shader: " << path.string() << std::endl;
                return words;
            }
        }
        throw std::runtime_error(std::string("Cannot find ") + filename +
            ". rebuild shaders with the lab 3 CMakeLists.txt");
    }

    VkShaderModule loadShaderModule(const std::vector<uint32_t>& code) {
        const VkShaderModuleCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size() * sizeof(uint32_t), .pCode = code.data(),
        };
        VkShaderModule result = VK_NULL_HANDLE;
        check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr,
                                   &result), "vkCreateShaderModule");
        return result;
    }
    void createPipeline(const char* fragmentFile, VkPipeline& output)
    {
        auto& context = graphics::internal::context;
        const auto vertexCode = readShader("textured.vert.spv");
        const auto fragmentCode = readShader(fragmentFile);

        VkShaderModule vertexModule = loadShaderModule(vertexCode);
        VkShaderModule fragmentModule = VK_NULL_HANDLE;

        try {
            fragmentModule = loadShaderModule(fragmentCode);

            VkPipelineShaderStageCreateInfo vertexStage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexModule,
                .pName = "main"
            };

            VkPipelineShaderStageCreateInfo fragmentStage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragmentModule,
                .pName = "main"
            };

            const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {
                vertexStage, fragmentStage
            };

            VkVertexInputBindingDescription vertexBinding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
            };

            const std::array<VkVertexInputAttributeDescription, 3> attributes = {{
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
            }};

            VkPipelineVertexInputStateCreateInfo vertexInput{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertexBinding,
                .vertexAttributeDescriptionCount =
                    static_cast<uint32_t>(attributes.size()),
                .pVertexAttributeDescriptions = attributes.data()
            };

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE
            };

            VkPipelineViewportStateCreateInfo viewportState{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1
            };

            VkPipelineRasterizationStateCreateInfo rasterizer{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .lineWidth = 1.0f
            };

            VkPipelineMultisampleStateCreateInfo multisampling{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
            };

            VkPipelineDepthStencilStateCreateInfo depthStencil{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE
            };

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlending{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment
            };

            const std::array<VkDynamicState, 2> dynamicStates = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };

            VkPipelineDynamicStateCreateInfo dynamicState{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                .pDynamicStates = dynamicStates.data()
            };

            const VkDescriptorSetLayout layouts[] = {scene_set_layout, model_set_layout};
            VkPipelineLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount = 2;
            layoutInfo.pSetLayouts = layouts;

            const VkPushConstantRange pushRange = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(EffectPush)};
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pushRange;
            if (!pipeline_layout && vkCreatePipelineLayout(
                context.device, &layoutInfo, nullptr,
                &pipeline_layout) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create pipeline layout");
            }

            VkGraphicsPipelineCreateInfo pipelineInfo{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = static_cast<uint32_t>(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertexInput,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = pipeline_layout,
                .renderPass = context.render_pass,
                .subpass = 0
            };

            std::cerr << "creating Lab 3 texture pipeline..." << std::endl;
            if (vkCreateGraphicsPipelines(
                context.device, VK_NULL_HANDLE, 1,
                &pipelineInfo, nullptr, &output) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline");
            }
        }
        catch (...) {
            if (fragmentModule != VK_NULL_HANDLE) {
                vkDestroyShaderModule(context.device, fragmentModule, nullptr);
            }
            vkDestroyShaderModule(context.device, vertexModule, nullptr);
            throw;
        }

        vkDestroyShaderModule(context.device, fragmentModule, nullptr);
        vkDestroyShaderModule(context.device, vertexModule, nullptr);
    }

    void drawObjectControls(ObjectTransform& object, int id, const char* title) {
        ImGui::PushID(id);
        if (ImGui::TreeNode(title)) {
            ImGui::DragFloat3("Position", glm::value_ptr(object.position), 0.02f);
            ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(object.rotation), 0.5f);
            ImGui::DragFloat3("Scale", glm::value_ptr(object.scale), 0.02f, 0.05f, 12.0f);
            ImGui::ColorEdit3("Albedo", glm::value_ptr(object.albedo));
            ImGui::ColorEdit3("Specular color", glm::value_ptr(object.specularColor));
            ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f);
            if (ImGui::BeginCombo("Texture", textures[object.textureIndex].name.c_str())) {
                for (size_t i = 0; i < textures.size(); ++i) {
                    if (ImGui::Selectable(textures[i].name.c_str(), object.textureIndex == int(i)))
                        object.textureIndex = static_cast<int>(i);
                }
                ImGui::EndCombo();
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    void drawPointLightControls() {
        ImGui::BeginDisabled(point_lights.size() >= max_point_count);
        if (ImGui::Button("Add point light")) point_lights.emplace_back();
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Remove all points")) point_lights.clear();
        for (size_t i = 0; i < point_lights.size();) {
            ImGui::PushID(static_cast<int>(i));
            const std::string title = "Point " + std::to_string(i + 1);
            bool remove = false;
            if (ImGui::TreeNode(title.c_str())) {
                auto& light = point_lights[i];
                ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.03f);
                ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
                remove = ImGui::Button("Remove point");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) point_lights.erase(point_lights.begin() + static_cast<ptrdiff_t>(i));
            else ++i;
        }
        ImGui::Text("Count: %zu. GPU storage limit: %u", point_lights.size(), max_point_count);
    }

    void drawSpotLightControls() {
        ImGui::BeginDisabled(spot_lights.size() >= max_spot_count);
        if (ImGui::Button("Add spotlight")) spot_lights.emplace_back();
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Remove all spots")) spot_lights.clear();
        for (size_t i = 0; i < spot_lights.size();) {
            ImGui::PushID(static_cast<int>(i));
            const std::string title = "Spotlight " + std::to_string(i + 1);
            bool remove = false;
            if (ImGui::TreeNode(title.c_str())) {
                auto& light = spot_lights[i];
                ImGui::Checkbox("Attach to camera", &light.attachedToCamera);
                ImGui::BeginDisabled(light.attachedToCamera);
                ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.03f);
                ImGui::DragFloat3("Direction (rays)", glm::value_ptr(light.direction), 0.02f);
                ImGui::EndDisabled();
                ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
                ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
                ImGui::SliderFloat("Inner half-angle", &light.innerAngle, 0.0f, 89.0f, "%.1f deg");
                ImGui::SliderFloat("Outer half-angle", &light.outerAngle, 0.0f, 89.0f, "%.1f deg");
                light.innerAngle = std::clamp(light.innerAngle, 0.0f, 89.0f);
                light.outerAngle = std::clamp(light.outerAngle, light.innerAngle, 89.0f);
                ImGui::TextWrapped("Inner <= outer. Equal angles give a hard edge.");
                remove = ImGui::Button("Remove spotlight");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) spot_lights.erase(spot_lights.begin() + static_cast<ptrdiff_t>(i));
            else ++i;
        }
        ImGui::Text("Count: %zu. GPU storage limit: %u", spot_lights.size(), max_spot_count);
    }

    void updateCameraInput(double deltaTime) {
        const auto& io = ImGui::GetIO();
        auto& camera = g_appState.camera;
        if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            camera.yaw += io.MouseDelta.x * 0.15f;
            camera.pitch = std::clamp(camera.pitch - io.MouseDelta.y * 0.15f, -89.0f, 89.0f);
        }
        camera.yaw = std::remainder(camera.yaw, 360.0f);
        if (io.WantCaptureKeyboard) return;
        const glm::vec3 forward = cameraForward();
        const glm::vec3 right = safeNormalize(glm::cross(forward, glm::vec3(0,1,0)),
                                               glm::vec3(1,0,0));
        glm::vec3 movement(0.0f);
        if (ImGui::IsKeyDown(ImGuiKey_W)) movement += forward;
        if (ImGui::IsKeyDown(ImGuiKey_S)) movement -= forward;
        if (ImGui::IsKeyDown(ImGuiKey_D)) movement += right;
        if (ImGui::IsKeyDown(ImGuiKey_A)) movement -= right;
        if (ImGui::IsKeyDown(ImGuiKey_E)) movement.y += 1.0f;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) movement.y -= 1.0f;
        if (glm::dot(movement, movement) > 0.0f)
            camera.position += glm::normalize(movement) * camera.movementSpeed * float(deltaTime);
    }

    void drawInterface() {
        ImGui::SetNextWindowSize(ImVec2(425, 650), ImGuiCond_FirstUseEver);
        ImGui::Begin("lab 3: Textures and sampling");
        ImGui::Text("Point lights: %zu | Spotlights: %zu", point_lights.size(), spot_lights.size());
        if (ImGui::Button("Only directional light")) {
            point_lights.clear();
            spot_lights.clear();
            g_appState.directionalEnabled = true;
            g_appState.directionalIntensity = 0.8f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset scene")) resetScene();

        if (ImGui::CollapsingHeader("Texture sampling", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Basic task: one texture, no effects")) {
                g_appState.cube1.textureIndex = g_appState.cube2.textureIndex = g_appState.floor.textureIndex = 0;
                g_appState.cube1.albedo = g_appState.cube2.albedo = g_appState.floor.albedo = glm::vec3(1);
                g_appState.effectEnabled = false;
                g_appState.basicMode = true;
            }
            if (ImGui::Checkbox("Animated sampling on Cube 1", &g_appState.effectEnabled))
                g_appState.basicMode = false;
            ImGui::Checkbox("Play texture effect", &g_appState.effectPlaying);
            ImGui::SliderFloat("Effect speed", &g_appState.effectSpeed, 0.0f, 3.0f);
            ImGui::SliderFloat("UV distortion", &g_appState.distortion, 0.0f, 0.3f);
            ImGui::SliderFloat("Multiple sample blend", &g_appState.sampleBlend, 0.0f, 1.0f);
            ImGui::SliderFloat("UV scale", &g_appState.uvScale, 0.2f, 8.0f);
            ImGui::ColorEdit3("Effect color modulation", glm::value_ptr(g_appState.effectTint));
            ImGui::SliderFloat("Emission", &g_appState.emission, 0.0f, 2.0f);
            if (ImGui::Button("Reset effect time")) g_appState.effectTime = 0.0;
        }
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto& camera = g_appState.camera;
            ImGui::DragFloat3("Camera position", glm::value_ptr(camera.position), 0.03f);
            ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.1f deg");
            if (ImGui::RadioButton("Perspective", g_appState.isPerspective))
                g_appState.isPerspective = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Orthographic", !g_appState.isPerspective))
                g_appState.isPerspective = false;
            if (g_appState.isPerspective)
                ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 100.0f, "%.1f deg");
            else
                ImGui::SliderFloat("Orthographic half-height", &g_appState.orthoSize, 1.0f, 15.0f);
            ImGui::SliderFloat("Movement speed", &camera.movementSpeed, 0.1f, 10.0f);
            if (ImGui::Button("Reset camera")) camera = Camera{};
            ImGui::TextWrapped("Outside UI: WASD move, Q/E down/up, hold right mouse to look.");
        }

        if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient color", glm::value_ptr(g_appState.ambientColor));
            ImGui::SliderFloat("Ambient intensity", &g_appState.ambientIntensity, 0.0f, 1.0f);
            ImGui::Checkbox("Directional enabled", &g_appState.directionalEnabled);
            ImGui::DragFloat3("Directional rays", glm::value_ptr(g_appState.directionalDirection), 0.02f);
            ImGui::ColorEdit3("Directional color", glm::value_ptr(g_appState.directionalColor));
            ImGui::SliderFloat("Directional intensity", &g_appState.directionalIntensity, 0.0f, 5.0f);
            ImGui::TextWrapped("Direction describes light rays. Shader uses its opposite as L.");
        }
        if (ImGui::CollapsingHeader("Models and materials", ImGuiTreeNodeFlags_DefaultOpen)) {
            drawObjectControls(g_appState.cube1, 0, "Cube 1");
            drawObjectControls(g_appState.cube2, 1, "Cube 2");
            ImGui::Checkbox("Show floor", &g_appState.showFloor);
            if (g_appState.showFloor) drawObjectControls(g_appState.floor, 2, "Floor");
        }
        if (ImGui::CollapsingHeader("Lab 1 animation")) {
            if (ImGui::Button(g_appState.isAnimating ? "Pause" : "Play")) {
                g_appState.isAnimating = !g_appState.isAnimating;
                if (g_appState.isAnimating) g_appState.animationStarted = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset animation")) {
                g_appState.isAnimating = false;
                g_appState.animationStarted = false;
                g_appState.animTime = 0.0;
            }
            ImGui::SliderFloat("Speed", &g_appState.animSpeed, 0.0f, 5.0f);
            ImGui::SliderFloat("Radius", &g_appState.animRadius, 0.0f, 4.0f);
            ImGui::SliderFloat("Height", &g_appState.animHeight, 0.0f, 2.0f);
            ImGui::SliderInt("Vertical waves", &g_appState.verticalWaves, 1, 6);
            ImGui::SliderFloat("Rotation speed", &g_appState.rotationSpeed, -180.0f, 180.0f);
        }
        ImGui::PushID("points");
        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen))
            drawPointLightControls();
        ImGui::PopID();
        ImGui::PushID("spots");
        if (ImGui::CollapsingHeader("Spotlights", ImGuiTreeNodeFlags_DefaultOpen))
            drawSpotLightControls();
        ImGui::PopID();
        ImGui::TextWrapped("Point/spot attenuation: 1 / max(distance^2, 0.01). No shadows in this lab.");
        ImGui::End();
    }

    glm::mat4 objectModel(const ObjectTransform& object, int animationSign) {
        glm::vec3 position = object.position;
        glm::vec3 rotation = object.rotation;
        if (animationSign != 0 && g_appState.animationStarted) {
            const double t = g_appState.animTime;
            position += float(animationSign) * glm::vec3(
                float(g_appState.animRadius * std::sin(t)),
                float(g_appState.animHeight * std::sin(g_appState.verticalWaves * t)),
                float(0.5 * g_appState.animRadius * std::sin(2.0 * t)));
            rotation.y += float(std::fmod(animationSign * t * g_appState.rotationSpeed, 360.0));
        }
        glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
        model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1,0,0));
        model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0,1,0));
        model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0,0,1));

        const glm::vec3 scale = glm::max(object.scale, glm::vec3(0.05f));
        return glm::scale(model, scale);
    }

    void updateModelUniforms(size_t index, const ObjectTransform& object, int sign) {
        ModelUniforms data{};
        data.model = objectModel(object, sign);
        data.normalMatrix = glm::transpose(glm::inverse(data.model));
        data.albedo = glm::vec4(object.albedo, 1.0f);
        data.specularColor = glm::vec4(object.specularColor, 1.0f);
        data.materialParameters = {std::max(object.shininess, 1.0f), 0, 0, 0};
        const VkDeviceSize offset = index * model_stride;
        std::memcpy(static_cast<unsigned char*>(model_uniform_buffer.mapped) + offset, &data, sizeof(data));
    }

    void updateGpuUniforms() {
        auto& context = graphics::internal::context;
        ensureLightCapacity(point_light_buffer, point_lights.size(), sizeof(GpuPointLight), 1);
        ensureLightCapacity(spot_light_buffer, spot_lights.size(), sizeof(GpuSpotLight), 2);
        std::vector<GpuPointLight> gpuPoints;
        gpuPoints.reserve(point_lights.size());
        for (const auto& light : point_lights)
            gpuPoints.push_back({glm::vec4(light.position, 1.0f),
                                 glm::vec4(light.color, std::max(light.intensity, 0.0f))});
        writeBuffer(point_light_buffer, gpuPoints.data(), gpuPoints.size() * sizeof(GpuPointLight));

        const glm::vec3 forward = cameraForward();
        std::vector<GpuSpotLight> gpuSpots;
        gpuSpots.reserve(spot_lights.size());
        for (const auto& light : spot_lights) {
            const auto position = light.attachedToCamera ? g_appState.camera.position : light.position;
            const auto direction = light.attachedToCamera ? forward :
                safeNormalize(light.direction, glm::vec3(0.0f, -1.0f, 0.0f));
            const float inner = std::clamp(light.innerAngle, 0.0f, 89.0f);
            const float outer = std::clamp(light.outerAngle, inner, 89.0f);
            gpuSpots.push_back({glm::vec4(position, 1.0f), glm::vec4(direction, 0.0f),
                glm::vec4(light.color, std::max(light.intensity, 0.0f)),
                glm::vec4(std::cos(glm::radians(inner)), std::cos(glm::radians(outer)), 0, 0)});
        }
        writeBuffer(spot_light_buffer, gpuSpots.data(), gpuSpots.size() * sizeof(GpuSpotLight));

        const auto& camera = g_appState.camera;
        const float aspect = context.swapchain_extent.height ?
            float(context.swapchain_extent.width) / float(context.swapchain_extent.height) : 1.0f;
        const glm::mat4 view = glm::lookAt(camera.position, camera.position + forward, glm::vec3(0,1,0));
        glm::mat4 projection;
        if (g_appState.isPerspective)
            projection = glm::perspective(glm::radians(std::clamp(camera.fov, 20.0f, 100.0f)), aspect, 0.1f, 100.0f);
        else {
            const float size = std::max(g_appState.orthoSize, 0.1f);
            projection = glm::ortho(-size * aspect, size * aspect, -size, size, 0.1f, 100.0f);
        }
        projection[1][1] *= -1.0f;
        SceneUniforms scene{};
        scene.viewProjection = projection * view;
        scene.cameraPosition = glm::vec4(camera.position, 1.0f);
        scene.ambientColorIntensity = glm::vec4(g_appState.ambientColor, std::max(g_appState.ambientIntensity, 0.0f));
        scene.directionalDirection = glm::vec4(safeNormalize(g_appState.directionalDirection,
                                                             glm::vec3(0,-1,0)), 0.0f);
        scene.directionalColorIntensity = glm::vec4(g_appState.directionalColor,
            g_appState.directionalEnabled ? std::max(g_appState.directionalIntensity, 0.0f) : 0.0f);
        scene.lightCounts = {static_cast<uint32_t>(point_lights.size()),
                             static_cast<uint32_t>(spot_lights.size()), 0, 0};
        writeBuffer(scene_uniform_buffer, &scene, sizeof(scene));
        updateModelUniforms(0, g_appState.cube1, 1);
        updateModelUniforms(1, g_appState.cube2, -1);
        if (g_appState.showFloor) updateModelUniforms(2, g_appState.floor, 0);
        check(vmaFlushAllocation(context.allocator, model_uniform_buffer.allocation,
                                 0, model_uniform_buffer.size), "flush dynamic UBO");
    }

    bool initialize() {
        try {
            resetScene();
            previous_time = -1.0;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
            max_storage_range = properties.limits.maxStorageBufferRange;
            max_point_count = static_cast<uint32_t>(max_storage_range / sizeof(GpuPointLight));
            max_spot_count = static_cast<uint32_t>(max_storage_range / sizeof(GpuSpotLight));
            if (!max_point_count || !max_spot_count)
                throw std::runtime_error("Storage buffer limits are too small");
            const VkDeviceSize alignment = std::max<VkDeviceSize>(1, properties.limits.minUniformBufferOffsetAlignment);
            model_stride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;
            if (sizeof(ModelUniforms) > properties.limits.maxUniformBufferRange ||
                2 * model_stride > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("Dynamic UBO limits exceeded");
            const VkCommandPoolCreateInfo pool = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = graphics::internal::context.graphics_queue_index,
            };
            check(vkCreateCommandPool(graphics::internal::context.device, &pool, nullptr, &upload_pool), "upload pool");
            createCube();
            loadTextures();
            createDescriptorResources();
            createPipeline("textured.frag.spv", pipeline);
            createPipeline("textured_effect.frag.spv", effect_pipeline);
            std::cerr << "Lab 3 ready: GPU geometry, six textures, dynamic UBO" << std::endl;
            return true;
        } catch (const std::exception& error) {
            std::cerr << "Lab 3 initialization failed: " << error.what() << std::endl;
            shutdown();
            return false;
        }
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        if (!context.device) return;
        vkDeviceWaitIdle(context.device);
        if (effect_pipeline) vkDestroyPipeline(context.device, effect_pipeline, nullptr);
        effect_pipeline = VK_NULL_HANDLE;
        if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
        if (scene_set_layout) vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
        if (model_set_layout) vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
        pipeline = VK_NULL_HANDLE;
        pipeline_layout = VK_NULL_HANDLE;
        descriptor_pool = VK_NULL_HANDLE;
        scene_set_layout = model_set_layout = VK_NULL_HANDLE;
        scene_descriptor_set = VK_NULL_HANDLE;
        destroyBuffer(model_uniform_buffer);
        for (auto& texture : textures) {
            if (texture.view) vkDestroyImageView(context.device, texture.view, nullptr);
            if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
        }
        textures.clear();
        if (texture_sampler) vkDestroySampler(context.device, texture_sampler, nullptr);
        if (upload_pool) vkDestroyCommandPool(context.device, upload_pool, nullptr);
        texture_sampler = VK_NULL_HANDLE;
        upload_pool = VK_NULL_HANDLE;
        destroyBuffer(scene_uniform_buffer);
        destroyBuffer(point_light_buffer);
        destroyBuffer(spot_light_buffer);
        destroyBuffer(index_buffer);
        destroyBuffer(vertex_buffer);
        cube_index_count = 0;
    }

    void update(double time) {
        const double deltaTime = previous_time < 0.0 ? 0.0 :
            std::clamp(time - previous_time, 0.0, 0.1);
        previous_time = time;
        if (g_appState.isAnimating) g_appState.animTime += deltaTime * g_appState.animSpeed;
        if (g_appState.effectPlaying) g_appState.effectTime += deltaTime * g_appState.effectSpeed;
        updateCameraInput(deltaTime);
        drawInterface();

    }

    void render(const graphics::internal::FrameData& fd) {
        auto& context = graphics::internal::context;
        if (!pipeline || !fd.command_buffer || !context.swapchain_extent.width ||
            !context.swapchain_extent.height) return;

        updateGpuUniforms();
        const VkViewport viewport = {
            .x = 0, .y = 0,
            .width = static_cast<float>(context.swapchain_extent.width),
            .height = static_cast<float>(context.swapchain_extent.height),
            .minDepth = 0, .maxDepth = 1,
        };
        const VkRect2D scissor = {.offset = {0,0}, .extent = context.swapchain_extent};
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);
        const size_t objectCount = g_appState.showFloor ? 3 : 2;
        const ObjectTransform* objects[] = {&g_appState.cube1, &g_appState.cube2, &g_appState.floor};
        for (size_t i = 0; i < objectCount; ++i) {
            const bool effect = i == 0 && !g_appState.basicMode;
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              effect ? effect_pipeline : pipeline);
            const uint32_t dynamicOffset = static_cast<uint32_t>(i * model_stride);
            const VkDescriptorSet material = textures[objects[i]->textureIndex].descriptor;
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_layout, 1, 1, &material, 1, &dynamicOffset);
            if (effect) {
                const EffectPush push = {
                    {float(g_appState.effectTime), g_appState.distortion, g_appState.sampleBlend, g_appState.uvScale},
                    glm::vec4(g_appState.effectTint, g_appState.emission),
                    {g_appState.effectEnabled ? 1u : 0u, 0u, 0u, 0u}
                };
                vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0, sizeof(push), &push);
            }
            vkCmdDrawIndexed(fd.command_buffer, cube_index_count, 1, 0, 0, 0);
        }

    }

}
