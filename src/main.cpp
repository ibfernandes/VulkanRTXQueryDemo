#define RESOURCES_DIR  "C:/path to where you downloaded the repo..."
#define VULKAN_SDK_PATH "C:/VulkanSDK/1.2.189.2/Bin32/glslc.exe --target-env=vulkan1.2 "
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_XYZW_ONLY
#define GLM_FORCE_LEFT_HANDED
#define MAX_FRAMES_IN_FLIGHT 2
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define VK_CHECK(f){\
	VkResult res = (f);\
	if (res != VK_SUCCESS){\
		std::cout << "VK FATAL ERROR : " << translateVulkanErrorCode(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n";\
		assert(res == VK_SUCCESS);\
	}\
}

GLFWwindow* window;
static const int WIDTH = 1200;
static const int HEIGHT = 600;

static std::string translateVulkanErrorCode(VkResult code) {
    switch (code) {
#define STR(r) case VK_##r: return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
    }
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

struct QueueFamilyIndicesVK {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

std::string loadShader(std::string filename) {
    std::ifstream file(RESOURCES_DIR + filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);


    file.close();

    return std::string(buffer.begin(), buffer.end());
}

void compileGLSLtoSPIRV(std::string shaderRelPath) {

    std::string vert = std::string(VULKAN_SDK_PATH) + "\"" + (RESOURCES_DIR + shaderRelPath) + ".vert" + "\"" + " -o " + "\"" + (RESOURCES_DIR + shaderRelPath) + ".vert.spv" + "\"";
    system(vert.c_str());

    std::string frag = std::string(VULKAN_SDK_PATH) + "\"" + (RESOURCES_DIR + shaderRelPath) + ".frag" + "\"" + " -o " + "\"" + (RESOURCES_DIR + shaderRelPath) + ".frag.spv" + "\"";
    system(frag.c_str());
}

VkDeviceAddress getDeviceAddress(VkDevice logicalDevice, VkBuffer buffer) {
    VkBufferDeviceAddressInfoKHR bufferDeviceAddresInfo{};
    bufferDeviceAddresInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddresInfo.buffer = buffer;
    return reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(logicalDevice, "vkGetBufferDeviceAddressKHR"))(logicalDevice, &bufferDeviceAddresInfo);
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return -1;
}

VkBuffer createBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkDeviceMemory *deviceMem, uint32_t memSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memFlags) {
    VkBuffer buffer;
    
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = memSize;
    bufferInfo.usage = usageFlags;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.flags = 0;

    VK_CHECK(vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memFlags);
    allocInfo.pNext = &memoryAllocateFlagsInfo;

    VK_CHECK(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, deviceMem));
    VK_CHECK(vkBindBufferMemory(logicalDevice, buffer, *deviceMem, 0));

    return buffer;
}

//Sample with no templaters or helps, just pure vulkan and c++.
int vulkan() {
    //Steam overlay bug fix
    _putenv("DISABLE_VK_LAYER_VALVE_steam_overlay_1=1");
    /*
        GLFW window management.
    */
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan RTX Demo", nullptr, nullptr);

    if (!glfwVulkanSupported()) 
        std::cout << "GLFW Vulkan not supported" << std::endl;
    
    /*
    *   Step 0
    *   Application data used during the program.
    *   Defines all geometries.
    */
    //Each vertex is made of 3 floating point numbers representing the coordinates (x,y,z)
    uint32_t vertexStride = sizeof(float) * 3;
    //Quad vertex data
    float quad[4 * 3] = { -0.5f, 0.5f, 0.0f,
                        0.5f, 0.5f, 0.0f,
                        0.5f, -0.5f, 0.0f,
                        -0.5f, -0.5f, 0.0f };
    uint32_t quadNVertices = 4;
    uint32_t quadNTriangles = 2;
    uint32_t quadIndices[6] = { 3, 1, 0,
                                3, 2 ,1 };
    

    // Cube indices mapping
    //   6 ---- 7
    //  /      / |
    // 2 ---- 3  |
    // |  4   |  5
    // | /    | /
    // 0 ---- 1
    //Cube vertex data
    float cube[8 * 3] = {  -0.5f, -0.5f, -0.5f,
                            0.5f, -0.5f, -0.5f,
                            0.5f, 0.5f, -0.5f,
                           -0.5f, 0.5f, -0.5f, 
                           -0.5f, -0.5f, 0.5f,
                            0.5f, -0.5f, 0.5f,
                            0.5f, 0.5f, 0.5f,
                           -0.5f, 0.5f, 0.5f };
    uint32_t cubeNTriangles = 6 * 2;
    uint32_t cubeNVertices = 8;
    uint32_t cubeIndices[6 * 2 * 3] = { 0,1,3,//back face
                                        1,2,3,
                                        5,1,2,//right face
                                        2,5,6,
                                        4,5,6,//front face
                                        6,4,7,
                                        4,0,3,//left face
                                        3,7,4,
                                        7,6,2,//top face
                                        2,3,7,
                                        4,5,1,//bottom face
                                        1,0,4};

    //Cornell box
    uint32_t cornellboxNTriangles = 32;
    uint32_t cornellboxNVertices = 28;
    float cornellbox[28 * 3] = { 
        1.010000, -0.000000, -0.990000,
        1.020000, 1.990000, -0.990000,
        1.020000, 1.990000, 1.040000,
        -1.000000, -0.000000, -0.990000,
        -1.000000, 0.000000, 1.040000,
        -0.530000, 0.600000, -0.750000,
        0.050000, 0.600000, -0.570000,
        -0.130000, 0.000000, 0.000000,
        0.050000, -0.000000, -0.570000,
        -0.130000, 0.600000, -0.000000,
        -0.530000, -0.000000, -0.750000,
        -0.700000, -0.000000, -0.170000,
        0.710000, 1.200000, 0.490000,
        0.530000, 1.200000, -0.090000,
        0.140000, 0.000000, 0.670000,
        0.710000, 0.000000, 0.490000,
        -0.040000, 0.000000, 0.090000,
        0.530000, -0.000000, -0.090000,
        0.240000, 1.980000, 0.220000,
        -0.230000, 1.980000, -0.160000,
        0.240000, 1.980000, -0.160000,
        -1.000000, 1.990000, 1.040000,
        -1.000000, 1.990000, -0.990000,
        0.990000, 0.000000, 1.040000,
        -0.700000, 0.600000, -0.170000,
        0.140000, 1.200000, 0.670000,
        -0.040000, 1.200000, 0.090000,
        -0.230000, 1.980000, 0.220000
    };

    uint32_t cornellboxIndices[32 * 3] = {
        4, 24, 1,
        3, 23, 2,
        24, 22, 3,
        4, 22, 5,
        1, 3, 2,
        6, 10, 7,
        7, 8, 9,
        6, 9, 11,
        10, 12, 8,
        25, 11, 12,
        27, 13, 14,
        14, 16, 18,
        13, 15, 16,
        26, 17, 15,
        27, 18, 17,
        19, 20, 21,
        4, 5, 24,
        3, 22, 23,
        24, 5, 22,
        4, 23, 22,
        1, 24, 3,
        6, 25, 10,
        7, 10, 8,
        6, 7, 9,
        10, 25, 12,
        25, 6, 11,
        27, 26, 13,
        14, 13, 16,
        13, 26, 15,
        26, 27, 17,
        27, 14, 18,
        19, 28, 20
    };
    //Shift by one. Obj file indices start at 1, instead of 0.
    for (int i = 0; i < 32 * 3; i++)
        cornellboxIndices[i] = cornellboxIndices[i] - 1;

    /*
        Program uniforms.
    */
    glm::vec3 camPosition = glm::vec3(0,0,-6);
    glm::vec3 up = glm::vec3(0,-1,0);
    glm::vec3 front = glm::vec3(0,0,1);
    float projAngle = 45;

    struct UniformBuffer {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec3 color;
    };

    UniformBuffer ubo{};
    ubo.color = glm::vec3(1,1,1);
    ubo.model = glm::mat4(1);
    ubo.model = glm::translate(ubo.model, glm::vec3(0,-1,0));
    ubo.view = glm::lookAt(camPosition, glm::vec3(0.01,0.01,0.01), up);
    ubo.proj = glm::perspective(glm::radians(projAngle), (GLfloat)WIDTH / (GLfloat)HEIGHT, 0.001f, 10000.0f);

    /*
        Vulkan variables roughly sorted by order of creation.
    */
    const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
    std::vector<const char*> deviceExtensions = {
        "VK_KHR_swapchain",
        "VK_KHR_ray_tracing_pipeline",
        "VK_KHR_ray_query",
        "VK_KHR_acceleration_structure",
        "VK_KHR_spirv_1_4",
        "VK_EXT_descriptor_indexing",
        "VK_KHR_buffer_device_address",
        "VK_KHR_deferred_host_operations"
    };

    int currentFrame = 0;
    const uint16_t nBackBuffers = 3;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
    VkQueue presentationQueue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    int nSwapChainImages = 0;
    VkImage swapChainImages[nBackBuffers] = {};
    VkImageView swapChainImageViews[nBackBuffers] = {};
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMem = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkFramebuffer pipelineFrameBuffers[nBackBuffers];
    VkCommandPool commandPool;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    VkDescriptorPool descriptorPool;

    const uint16_t nVertexBuffers = 1;
    VkDeviceMemory vertexDeviceMem;
    void* vertexHostMem;
    VkBuffer vertexBuffer;

    VkDeviceMemory indexDeviceMem;
    void* indexHostMem;
    VkBuffer indexBuffer;

    VkDeviceMemory blasDeviceMem;
    VkBuffer blasBuffer;
    VkAccelerationStructureKHR blasAS;
    VkDeviceMemory blasScratchDeviceMem;
    VkBuffer blasScratchBuffer;
    VkDeviceAddress blasDeviceAddress;

    VkDeviceMemory tlasDeviceMem;
    VkBuffer tlasBuffer;
    VkBuffer tlasScratchBuffer;
    VkDeviceMemory tlasScratchDeviceMem;
    VkAccelerationStructureKHR tlasAS;
    VkDeviceAddress tlasDeviceAddress;

    const uint16_t nDescriptorSets = 2;
    VkDescriptorSet descriptorSets[nDescriptorSets];

    VkCommandBuffer commandBuffers[nBackBuffers];
    /*
        ---------------------------------------------------------------------
        We start by creating a Vulkan Instance with validation layers enabled.
        Validation layers add debugger capabilities to our program.
    */
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Ray Query Demo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2; //Ray tracing requires Vulkan v1.1
                                             //and vkDeviceBufferAddress requires Vulkan v1.2.
    //Retrive GLFW extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back("VK_EXT_debug_utils");

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo{};
    debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugUtilsCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugUtilsCreateInfo.pfnUserCallback = debugCallback;
    debugUtilsCreateInfo.pUserData = nullptr;
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    /*
        We use GLFW to create a surface using the vulkan instace we just created.
    */
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    /*
        We must select a physical device
    */
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    //There is no physical GPU device in this computer
    if (deviceCount == 0)
        return 0;

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    //For simplicity's sake we select the first physical GPU device available.
    physicalDevice = devices[0];

    /*
        Next we query for available queues in this physical device.
    */
    QueueFamilyIndicesVK indices;

    //First we retrieve the queues avaiable using vkGetPhysicalDeviceQueueFamilyProperties.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    //For each queue available we check for the desired property.
    for(int i = 0; i < queueFamilies.size(); i++){
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
        //We check if this physical device has a queue family capable of graphic rendering in the screen.
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;
        if (presentSupport)
            indices.presentFamily = i;

        if (indices.isComplete())
            break;
    }

    /*
        ---------------------------------------------------------------------
        We create a logical device
    */
    //We must enable 4 features in order to operate with ray tracing in Vulkan using ray query.
    // RayQuery, RayTracingPipeline, DeviceAddress and Acceleration Structure.
    VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures{};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR enabledRayTracingPipelineFeatures{};
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR enabledBufferDeviceAddresFeatures{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{};
    // Enable features required for ray tracing using feature chaining via pNext		
    enabledBufferDeviceAddresFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    enabledBufferDeviceAddresFeatures.bufferDeviceAddress = VK_TRUE;
    enabledRayTracingPipelineFeatures.pNext = nullptr;

    enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddresFeatures;

    enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
    enabledAccelerationStructureFeatures.pNext = &enabledRayTracingPipelineFeatures;

    enabledRayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    enabledRayQueryFeatures.rayQuery = VK_TRUE;
    enabledRayQueryFeatures.pNext = &enabledAccelerationStructureFeatures;

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
    createInfo.pNext = &enabledRayQueryFeatures; // Ray tracing features must be set via pNext

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice));
    /*
        Get queues from logicalDevice
    */

    vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentationQueue);
    vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
    /*
    * SwapChain
    */
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> surfaceFormats;
    std::vector<VkPresentModeKHR> presentModes;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        surfaceFormats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
    }

    VkSurfaceFormatKHR surfaceFormat{};
    for (const VkSurfaceFormatKHR& availableFormat : surfaceFormats)
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            surfaceFormat = availableFormat;

    VkPresentModeKHR presentMode{};
    for (const VkPresentModeKHR& availablePresentMode : presentModes)
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            presentMode = availablePresentMode; // VK_PRESENT_MODE_FIFO_KHR
    
    //Set swapChainExtent
    if (capabilities.currentExtent.width != UINT32_MAX)
        swapChainExtent = capabilities.currentExtent;
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        swapChainExtent =  actualExtent;
    }

    uint32_t imageCount = capabilities.minImageCount + 1;
    nSwapChainImages = imageCount;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
        imageCount = capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainCreateInfo.surface = surface;
    swapChainCreateInfo.minImageCount = imageCount;
    swapChainCreateInfo.imageFormat = surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapChainCreateInfo.queueFamilyIndexCount = 0;
    swapChainCreateInfo.pQueueFamilyIndices = nullptr;
    swapChainCreateInfo.preTransform = capabilities.currentTransform;
    swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainCreateInfo.presentMode = presentMode;
    swapChainCreateInfo.clipped = VK_TRUE;
    swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(logicalDevice, &swapChainCreateInfo, nullptr, &swapChain))

    vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
    vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages);
    swapChainImageFormat = surfaceFormat.format;

    for (size_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.image = swapChainImages[i];
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = swapChainImageFormat;
        imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(logicalDevice, &imageViewCreateInfo, nullptr, &swapChainImageViews[i]));
    }

    /*
        Create Depth Buffer
     */
     //Depth buffer image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapChainExtent.width;
    imageInfo.extent.height = swapChainExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_D24_UNORM_S8_UINT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateImage(logicalDevice, &imageInfo, nullptr, &depthImage));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(logicalDevice, depthImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &depthImageMem));
    VK_CHECK(vkBindImageMemory(logicalDevice, depthImage, depthImageMem, 0));

    //Depth buffer image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D24_UNORM_S8_UINT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(logicalDevice, &viewInfo, nullptr, &depthImageView));

    /*
        Create render Pass
    */

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D24_UNORM_S8_UINT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;


    std::vector<VkAttachmentDescription> attachments = { colorAttachment, depthAttachment };
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass));

    /*
        Vulkan ray tracing function pointers.
    */
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
     vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(logicalDevice, "vkGetBufferDeviceAddressKHR"));
     vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(logicalDevice, "vkCmdBuildAccelerationStructuresKHR"));
     vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(logicalDevice, "vkBuildAccelerationStructuresKHR"));
     vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(logicalDevice, "vkCreateAccelerationStructureKHR"));
     vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(logicalDevice, "vkDestroyAccelerationStructureKHR"));
     vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureBuildSizesKHR"));
     vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(logicalDevice, "vkGetAccelerationStructureDeviceAddressKHR"));
     vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(logicalDevice, "vkCmdTraceRaysKHR"));
     vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(logicalDevice, "vkGetRayTracingShaderGroupHandlesKHR"));
     vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(logicalDevice, "vkCreateRayTracingPipelinesKHR"));

    /*
        Compile shaders
    */
    compileGLSLtoSPIRV("shadowQuery");
    std::string vertShaderData = loadShader("shadowQuery.vert.spv");
    std::string fragShaderData = loadShader("shadowQuery.frag.spv");
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;

    VkShaderModuleCreateInfo vertShaderModulecreateInfo{};
    vertShaderModulecreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertShaderModulecreateInfo.codeSize = vertShaderData.size();
    vertShaderModulecreateInfo.pCode = reinterpret_cast<const uint32_t*>(vertShaderData.data());
    VK_CHECK(vkCreateShaderModule(logicalDevice, &vertShaderModulecreateInfo, nullptr, &vertShaderModule));

    VkShaderModuleCreateInfo fragShaderModulecreateInfo{};
    fragShaderModulecreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fragShaderModulecreateInfo.codeSize = fragShaderData.size();
    fragShaderModulecreateInfo.pCode = reinterpret_cast<const uint32_t*>(fragShaderData.data());
    VK_CHECK(vkCreateShaderModule(logicalDevice, &fragShaderModulecreateInfo, nullptr, &fragShaderModule));

    /*
        Configure descriptorSetLayout
    */
    int bindingCount = 2;
    VkDescriptorSetLayoutBinding layoutBindingds[2];
    VkDescriptorSetLayout descriptorSetLayout;

    layoutBindingds[0] = {};
    layoutBindingds[0].binding = 0;
    layoutBindingds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindingds[0].descriptorCount = 1;
    layoutBindingds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; //TODO: For now set on all stages

    layoutBindingds[1] = {};
    layoutBindingds[1].binding = 1;
    layoutBindingds[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    layoutBindingds[1].descriptorCount = 1;
    layoutBindingds[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; //TODO: For now set on all stages

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings = &layoutBindingds[0];

    VK_CHECK(vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout));
    
    /*
        Create Pipeline
    */
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkVertexInputBindingDescription bindDesc{};
    bindDesc.binding = 0;
    bindDesc.stride = vertexStride;
    bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    const uint16_t nAttributes = 1;
    //draw in text a data example layout
    VkVertexInputAttributeDescription attrDesc[nAttributes]{};
    attrDesc[0].binding = 0; //TODO dif between binding and location
    attrDesc[0].location = 0;
    attrDesc[0].format = VK_FORMAT_R32G32B32_SFLOAT; // Each vertex is made of three 32 bits floating point numbers 
    attrDesc[0].offset = 0;  //offset into the data buffer

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(nAttributes);
    vertexInputInfo.pVertexAttributeDescriptions = &attrDesc[0];

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = swapChainExtent.width;
    viewport.height = swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VK_CHECK(vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.pDepthStencilState = &depthStencil;

    VK_CHECK(vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    /*
        Create swapchain framebuffers
    */

    for (size_t i = 0; i < nSwapChainImages; i++) {

        std::vector<VkImageView> imageViewAttachments = {
            swapChainImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = imageViewAttachments.size();
        framebufferInfo.pAttachments = imageViewAttachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        VK_CHECK(vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &pipelineFrameBuffers[i]));
    }

    /*
        Create Command pools
    */
    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = indices.graphicsFamily.value();

    VK_CHECK(vkCreateCommandPool(logicalDevice, &commandPoolInfo, nullptr, &commandPool));

    /*
        Create semaphore and fences
    */
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(nSwapChainImages, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < 2; i++) {
        VK_CHECK(vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]));
        VK_CHECK(vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]));
    }

    /*
        Create descriptor pool
    */

    int poolSizeCount = 2;
    VkDescriptorPoolSize poolsSize[2];
    poolsSize[0] = {};
    poolsSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolsSize[0].descriptorCount = static_cast<uint32_t>(nSwapChainImages);

    poolsSize[1] = {};
    poolsSize[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolsSize[1].descriptorCount = static_cast<uint32_t>(nSwapChainImages);

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = poolSizeCount;
    descriptorPoolInfo.pPoolSizes = &poolsSize[0];
    descriptorPoolInfo.maxSets = static_cast<uint32_t>(nSwapChainImages);

    VK_CHECK(vkCreateDescriptorPool(logicalDevice, &descriptorPoolInfo, nullptr, &descriptorPool));


    /*
        Create Vertex Buffer
    */
    vertexBuffer = createBuffer(physicalDevice, logicalDevice, &vertexDeviceMem, sizeof(cornellbox),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(logicalDevice, vertexDeviceMem, 0, sizeof(cornellbox), 0, &vertexHostMem);
    memcpy(vertexHostMem, &cornellbox[0], sizeof(cornellbox));

    /*
        Create Index Buffer
    */
    indexBuffer = createBuffer(physicalDevice, logicalDevice, &indexDeviceMem, sizeof(cornellboxIndices),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(logicalDevice, indexDeviceMem, 0, sizeof(cornellboxIndices), 0, &indexHostMem);
    memcpy(indexHostMem, &cornellboxIndices[0], sizeof(cornellboxIndices));

    /*
        Create Ray tracing Bottom Level Acceleration Structure (BLAS)
    */

    VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    accelerationStructureGeometry.geometry.triangles.vertexData.deviceAddress = getDeviceAddress(logicalDevice, vertexBuffer);
    accelerationStructureGeometry.geometry.triangles.maxVertex = cornellboxNVertices;
    accelerationStructureGeometry.geometry.triangles.vertexStride = vertexStride;
    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    accelerationStructureGeometry.geometry.triangles.indexData.deviceAddress = getDeviceAddress(logicalDevice, indexBuffer);
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;

    // Get size info
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = 1;
    accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

    //Create Acceleration Structure buffer
    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};

    accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR( logicalDevice, 
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &accelerationStructureBuildGeometryInfo,
        &cornellboxNTriangles,
        &accelerationStructureBuildSizesInfo);

    blasBuffer = createBuffer(physicalDevice, logicalDevice, &blasDeviceMem, 
        accelerationStructureBuildSizesInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    blasScratchBuffer = createBuffer(physicalDevice, logicalDevice, &blasScratchDeviceMem,
        accelerationStructureBuildSizesInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // Acceleration structure
    VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
    accelerationStructureCreate_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreate_info.buffer = blasBuffer;
    accelerationStructureCreate_info.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
    accelerationStructureCreate_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    
    vkCreateAccelerationStructureKHR(logicalDevice, &accelerationStructureCreate_info, nullptr, &blasAS);

    VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
    accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationDeviceAddressInfo.accelerationStructure = blasAS;
    blasDeviceAddress = vkGetAccelerationStructureDeviceAddressKHR(logicalDevice, &accelerationDeviceAddressInfo);

    VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
    accelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationBuildGeometryInfo.dstAccelerationStructure = blasAS;
    accelerationBuildGeometryInfo.geometryCount = 1;
    accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
    accelerationBuildGeometryInfo.scratchData.deviceAddress = getDeviceAddress(logicalDevice, blasScratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount = cornellboxNTriangles;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    // Build the acceleration structure on the device via a one-time command buffer submission
    VkCommandBuffer commandBuffer{};

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1,
        &accelerationBuildGeometryInfo,
        accelerationBuildStructureRangeInfos.data());

    //Flush command buffer
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo blasFenceInfo{};
    blasFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    blasFenceInfo.flags = 0;

    VkFence blasFence;
    VK_CHECK(vkCreateFence(logicalDevice, &blasFenceInfo, nullptr, &blasFence));
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, blasFence));

    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(logicalDevice, 1, &blasFence, VK_TRUE, 100000000000));
    vkDestroyFence(logicalDevice, blasFence, nullptr);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);

    /*
        Create Ray tracing Top Level Acceleration Structure (TLAS)
    */
    VkTransformMatrixKHR transformMatrix = {
           1.0f, 0.0f, 0.0f, 0.0f,
           0.0f, 1.0f, 0.0f, 0.0f,
           0.0f, 0.0f, 1.0f, 0.0f };

    VkAccelerationStructureInstanceKHR tlasAccelerationStructureinstance{};
    tlasAccelerationStructureinstance.transform = transformMatrix;
    tlasAccelerationStructureinstance.instanceCustomIndex = 0;
    tlasAccelerationStructureinstance.mask = 0xFF; //The instance may only be hit if Cull Mask & instance.mask != 0
    tlasAccelerationStructureinstance.instanceShaderBindingTableRecordOffset = 0;
    tlasAccelerationStructureinstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    tlasAccelerationStructureinstance.accelerationStructureReference = blasDeviceAddress;
    
    VkBuffer instancesBuffer;
    VkDeviceMemory instancesDeviceMemory;
    void* instancesHostMem;

    instancesBuffer = createBuffer(physicalDevice, logicalDevice, &instancesDeviceMemory,
        sizeof(VkAccelerationStructureInstanceKHR), 
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(logicalDevice, instancesDeviceMemory, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &instancesHostMem);
    memcpy(instancesHostMem, &tlasAccelerationStructureinstance, sizeof(VkAccelerationStructureInstanceKHR));

    VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
    instanceDataDeviceAddress.deviceAddress = getDeviceAddress(logicalDevice, instancesBuffer);

    VkAccelerationStructureGeometryKHR tlasAccelerationStructureGeometry{};
    tlasAccelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasAccelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasAccelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasAccelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasAccelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasAccelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

    // Get size info
    VkAccelerationStructureBuildGeometryInfoKHR tlasAccelerationStructureBuildGeometryInfo{};
    tlasAccelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasAccelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasAccelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasAccelerationStructureBuildGeometryInfo.geometryCount = 1;
    tlasAccelerationStructureBuildGeometryInfo.pGeometries = &tlasAccelerationStructureGeometry;

    uint32_t primitiveCount = 1; //TODO equals geometry/model count?

    VkAccelerationStructureBuildSizesInfoKHR tlasAccelerationStructureBuildSizesInfo{};
    tlasAccelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        logicalDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &tlasAccelerationStructureBuildGeometryInfo,
        &primitiveCount, //is a pointer to an array of pBuildInfo->geometryCount uint32_t values defining the number of primitives built into each geometry.
        &tlasAccelerationStructureBuildSizesInfo);

   tlasBuffer = createBuffer(physicalDevice, logicalDevice, &tlasDeviceMem,
       tlasAccelerationStructureBuildSizesInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    tlasScratchBuffer = createBuffer(physicalDevice, logicalDevice, &tlasScratchDeviceMem,
        tlasAccelerationStructureBuildSizesInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);;

    // Acceleration structure
    VkAccelerationStructureCreateInfoKHR tlasAccelerationStructureCreateInfo{};
    tlasAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasAccelerationStructureCreateInfo.buffer = tlasBuffer;
    tlasAccelerationStructureCreateInfo.size = tlasAccelerationStructureBuildSizesInfo.accelerationStructureSize;
    tlasAccelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(logicalDevice, &tlasAccelerationStructureCreateInfo, nullptr, &tlasAS);

    VkAccelerationStructureBuildGeometryInfoKHR tlasAccelerationBuildGeometryInfo{};
    tlasAccelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasAccelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasAccelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasAccelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasAccelerationBuildGeometryInfo.dstAccelerationStructure = tlasAS;
    tlasAccelerationBuildGeometryInfo.geometryCount = 1;
    tlasAccelerationBuildGeometryInfo.pGeometries = &tlasAccelerationStructureGeometry;
    tlasAccelerationBuildGeometryInfo.scratchData.deviceAddress = getDeviceAddress(logicalDevice, tlasScratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR  tlasAccelerationStructureBuildRangeInfo{};
    tlasAccelerationStructureBuildRangeInfo.primitiveCount = primitiveCount; //TODO always 1?
    tlasAccelerationStructureBuildRangeInfo.primitiveOffset = 0;
    tlasAccelerationStructureBuildRangeInfo.firstVertex = 0;
    tlasAccelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> tlasAccelerationBuildStructureRangeInfos = { &tlasAccelerationStructureBuildRangeInfo };

    // Build the acceleration structure on the device via a one-time command buffer submission
    VkCommandBufferAllocateInfo tlasCommandBufferAllocateInfo{};
    tlasCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    tlasCommandBufferAllocateInfo.commandPool = commandPool;
    tlasCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    tlasCommandBufferAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer tlasCommandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(logicalDevice, &tlasCommandBufferAllocateInfo, &tlasCommandBuffer));

    VkCommandBufferBeginInfo tlasCmdBufInfo{};
    tlasCmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(tlasCommandBuffer, &tlasCmdBufInfo));

    vkCmdBuildAccelerationStructuresKHR(
        tlasCommandBuffer,
        1,
        &tlasAccelerationBuildGeometryInfo,
        tlasAccelerationBuildStructureRangeInfos.data());

    //Flush command buffer
    VK_CHECK(vkEndCommandBuffer(tlasCommandBuffer));

    VkSubmitInfo tlasSubmitInfo{};
    tlasSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    tlasSubmitInfo.commandBufferCount = 1;
    tlasSubmitInfo.pCommandBuffers = &tlasCommandBuffer;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo tlasFenceInfo{};
    tlasFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    tlasFenceInfo.flags = 0;

    VkFence tlasFence;
    VK_CHECK(vkCreateFence(logicalDevice, &tlasFenceInfo, nullptr, &tlasFence));
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &tlasSubmitInfo, tlasFence));

    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(logicalDevice, 1, &tlasFence, VK_TRUE, 100000000000));
    vkDestroyFence(logicalDevice, tlasFence, nullptr);
    vkFreeCommandBuffers(logicalDevice, commandPool, 1, &tlasCommandBuffer);

    /*
        Create Uniform Buffer
    */
    VkBuffer uboBuffer;
    VkDeviceMemory uboDeviceMemory;
    void* uboData;
    uboBuffer = createBuffer(physicalDevice, logicalDevice, &uboDeviceMemory,
        sizeof(ubo), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(logicalDevice, uboDeviceMemory, 0, sizeof(ubo), 0, &uboData);
    memcpy(uboData, &ubo, sizeof(ubo));
    vkUnmapMemory(logicalDevice, uboDeviceMemory);

    /*
        Create Descriptor Sets
    */
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocInfo.descriptorPool = descriptorPool;
    descriptorSetAllocInfo.descriptorSetCount = 1;
    descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;

    VK_CHECK(vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocInfo, &descriptorSets[0]));

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.offset = 0;
    bufferInfo.buffer = uboBuffer;
    bufferInfo.range = sizeof(ubo);

    VkWriteDescriptorSet descriptorsWrite[nDescriptorSets];

    descriptorsWrite[0] = {};
    descriptorsWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorsWrite[0].dstSet = descriptorSets[0];
    descriptorsWrite[0].dstBinding = 0;
    descriptorsWrite[0].dstArrayElement = 0;
    descriptorsWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorsWrite[0].descriptorCount = 1;
    descriptorsWrite[0].pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(logicalDevice, 1, &descriptorsWrite[0], 0, nullptr);

    VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
    descriptorAccelerationStructureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
    descriptorAccelerationStructureInfo.pAccelerationStructures = &tlasAS;

    descriptorsWrite[1] = {};
    descriptorsWrite[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorsWrite[1].dstSet = descriptorSets[0];
    descriptorsWrite[1].dstBinding = 1;
    descriptorsWrite[1].descriptorCount = 1;
    descriptorsWrite[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    descriptorsWrite[1].pNext = &descriptorAccelerationStructureInfo; // The specialized acceleration structure descriptor has to be chained

    vkUpdateDescriptorSets(logicalDevice, 2, &descriptorsWrite[0], 0, nullptr);

    /*
        Create Command Buffers
    */
    VkCommandBufferAllocateInfo commandBufferAllocInfo{};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandBufferCount = nBackBuffers;

    VK_CHECK(vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocInfo, commandBuffers));

    //TODO put everything in a huge buffer
    for (size_t i = 0; i < nBackBuffers; i++) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VK_CHECK(vkBeginCommandBuffer(commandBuffers[i], &beginInfo));

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = pipelineFrameBuffers[i];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearValues[2]{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = 2;
        renderPassInfo.pClearValues = &clearValues[0];

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        for (int k = 0; k < nVertexBuffers; k++) {
            VkBuffer vertexBuffers[] = { vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        vkCmdBindDescriptorSets(commandBuffers[i], 
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

        //indices.size()
        vkCmdDrawIndexed(commandBuffers[i], cornellboxNTriangles * 3, 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffers[i]);

        VK_CHECK(vkEndCommandBuffer(commandBuffers[i]));
    }

    //Main loop
    while (!glfwWindowShouldClose(window)) {
        { //Render
            VK_CHECK(vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX));

            uint32_t imageIndex;
            VK_CHECK(vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex));

            // Check if a previous frame is using this image (i.e. there is its fence to wait on)
            if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) 
                VK_CHECK(vkWaitForFences(logicalDevice, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX));

            // Mark the image as now being in use by this frame
            imagesInFlight[imageIndex] = inFlightFences[currentFrame];

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

            VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            VK_CHECK(vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]));
            VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            VkSwapchainKHR swapChains[] = { swapChain };
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;

            VK_CHECK(vkQueuePresentKHR(presentationQueue, &presentInfo));
            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
        glfwPollEvents();
    }


    //Cleanup of vulkan variables
    vkDestroyAccelerationStructureKHR(logicalDevice, blasAS, nullptr);
    vkDestroyAccelerationStructureKHR(logicalDevice, tlasAS, nullptr);

    vkDestroyBuffer(logicalDevice, uboBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, tlasBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, tlasScratchBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, blasBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, blasScratchBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, instancesBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
    vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);

    vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
        //vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
        //vkDestroyFence(logicalDevice, imagesInFlight[i], nullptr);
    }

    vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

    for (int i = 0; i < nBackBuffers; i++)
        vkDestroyFramebuffer(logicalDevice, pipelineFrameBuffers[i], nullptr);

    vkDestroyPipeline(logicalDevice, pipeline, nullptr);
    
    vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
    
    vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
    
    vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);

    vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

    for (int i = 0; i < nBackBuffers; i ++) 
        vkDestroyImageView(logicalDevice, swapChainImageViews[i], nullptr);

    vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
    vkDestroyDevice(logicalDevice, nullptr);
    vkDestroyInstance(instance, nullptr);
}

int main(){
    vulkan();
    return 0;
}