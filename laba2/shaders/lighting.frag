#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambientColorIntensity;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParameters; // x: shininess, y: procedural colors enabled
} object;

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 position;
    vec4 direction; // light -> surface
    vec4 colorIntensity;
    vec4 coneAngles; // cos(inner), cos(outer)
};

// Runtime-sized arrays, not fixed-size arrays and not arrays of descriptors.
layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointLights;
layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotLights;

vec3 safeNormalize(vec3 value, vec3 fallback) {
    float squaredLength = dot(value, value);
    return squaredLength > 1e-10 ? value * inversesqrt(squaredLength) : fallback;
}

// Shared Blinn-Phong model for every type of direct light.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 baseColor) {
    float diffuseFactor = max(dot(N, L), 0.0);
    if (diffuseFactor <= 0.0) return vec3(0.0);
    vec3 halfway = L + V;
    float specularFactor = 0.0;
    // Opposite L and V must not produce normalize(vec3(0)).
    if (dot(halfway, halfway) > 1e-10) {
        vec3 H = normalize(halfway);
        specularFactor = pow(max(dot(N, H), 0.0),
                             max(object.materialParameters.x, 1.0));
    }
    vec3 diffuse = baseColor * diffuseFactor;
    vec3 specular = object.specularColor.rgb * specularFactor;
    return (diffuse + specular) * radiance;
}

float spotlightWeight(float cosTheta, float cosInner, float cosOuter) {
    // CPU maintains inner <= outer, hence cosInner >= cosOuter.
    if (cosInner <= cosOuter)
        return step(cosOuter, cosTheta); // equal angles: hard edge, no division by zero
    return smoothstep(cosOuter, cosInner, cosTheta);
}

void main() {
    vec3 N = safeNormalize(worldNormal, vec3(0.0, 1.0, 0.0));
    vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition, N);
    vec3 baseColor = object.albedo.rgb;
    if (object.materialParameters.y > 0.5) baseColor *= vertexColor;

    // Ambient is applied once per fragment, not once per source.
    vec3 color = baseColor * scene.ambientColorIntensity.rgb *
                 scene.ambientColorIntensity.a;

    vec3 directionalL = safeNormalize(-scene.directionalDirection.xyz, vec3(0,1,0));
    color += blinnPhong(N, V, directionalL,
        scene.directionalColorIntensity.rgb * scene.directionalColorIntensity.a, baseColor);

    // Descriptor range may include spare capacity. lightCounts is the active length.
    uint pointCount = min(scene.lightCounts.x, uint(pointLights.lights.length()));
    for (uint i = 0u; i < pointCount; ++i) {
        PointLight light = pointLights.lights[i];
        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.01);
        vec3 L = safeNormalize(toLight, N);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a / distanceSquared;
        color += blinnPhong(N, V, L, radiance, baseColor);
    }

    uint spotCount = min(scene.lightCounts.y, uint(spotLights.lights.length()));
    for (uint i = 0u; i < spotCount; ++i) {
        SpotLight light = spotLights.lights[i];
        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.01);
        vec3 L = safeNormalize(toLight, N);
        vec3 D = safeNormalize(light.direction.xyz, vec3(0,-1,0));
        float coneWeight = spotlightWeight(dot(D, -L), light.coneAngles.x, light.coneAngles.y);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a *
                        coneWeight / distanceSquared;
        color += blinnPhong(N, V, L, radiance, baseColor);
    }
    outColor = vec4(color, 1.0);
}
