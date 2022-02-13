set(CMAKE_GENERATOR_PLATFORM "x64")
cmake_minimum_required(VERSION 3.9)
project(VulkanRTXDemo)

include_directories(${CMAKE_BINARY_DIR}/src)

# Using cpp17
set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

# GLFW
add_subdirectory(includes/glfw)
include_directories(includes/glfw/include)

# GLM
include_directories(includes/glm)

# Sources files
set(_src_root_path "${VulkanRTXDemo_SOURCE_DIR}/src/")
file(
    GLOB_RECURSE SOURCE_FILES 
    LIST_DIRECTORIES false
    "${_src_root_path}/*.c*"
    "${_src_root_path}/*.h*"
)

foreach(_source IN ITEMS ${SOURCE_FILES})
    get_filename_component(_source_path "${_source}" PATH)
    file(RELATIVE_PATH _source_path_rel "${_src_root_path}" "${_source_path}")
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_group_path}" FILES "${_source}")
endforeach()

include_directories( "${CMAKE_SOURCE_DIR}/src" )

add_executable(VulkanRTXDemo ${SOURCE_FILES})
target_link_libraries(VulkanRTXDemo ${core_libs} glfw Vulkan::Vulkan)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT VulkanRTXDemo)