#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

layout (binding = 1, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) in vec3 fragColor;
layout(location = 4) in vec3 inWorldPos;
layout(location = 5) in vec3 modelPos;

layout(location = 0) out vec4 outColor;

void main() {	
	vec3 lightPos = vec3(-1,1,-4);
	vec3 toLight = lightPos - modelPos;
	
	rayQueryEXT rayQuery;
	rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF,
							modelPos, 0.00001, toLight, 1000.0);

	// Start the ray traversal, rayQueryProceedEXT returns false if the traversal is complete
	while (rayQueryProceedEXT(rayQuery)) { 
	}

	//Simplistic fade off effect w.r.t. the light source distance.
	outColor = vec4((1.0f / (length(lightPos - inWorldPos)*2)) * fragColor, 1.0f) ;

	// If the intersection has hit a triangle, the fragment is shadowed
	if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT ) 
		outColor = vec4(outColor.xyz * 0.4f, 1);
}
