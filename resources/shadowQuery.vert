#version 460

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragColor;
layout(location = 4) out vec3 outWorldPos;
layout(location = 5) out vec3 outModelPos;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;


void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
	outWorldPos = vec3(ubo.model * vec4(inPosition, 1.0));
    outModelPos = inPosition;
    fragColor = ubo.color;
}
