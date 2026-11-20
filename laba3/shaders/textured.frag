#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 textureUV;
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
    vec4 materialParameters; 
} object;

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};
struct SpotLight {
    vec4 position;
    vec4 direction; 
    vec4 colorIntensity;
    vec4 coneAngles; 
};


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


vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 baseColor) {
    float diffuseFactor = max(dot(N, L), 0.0);
    if (diffuseFactor <= 0.0) return vec3(0.0);
    vec3 halfway = L + V;
    float specularFactor = 0.0;
    
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
    
    if (cosInner <= cosOuter)
        return step(cosOuter, cosTheta); 
    return smoothstep(cosOuter, cosInner, cosTheta);
}

layout(set = 1, binding = 1) uniform sampler2D albedoTexture;
#ifdef ENABLE_EFFECTS
layout(push_constant) uniform EffectPush {
    vec4 animation;
    vec4 modulation;
    uvec4 control;
} effect;
#endif
vec3 sampleAlbedo(vec2 uv) {
#ifdef ENABLE_EFFECTS
    if (effect.control.x == 0u) return texture(albedoTexture, uv).rgb;
    float t = effect.animation.x;
    vec2 p = uv * effect.animation.w;
    vec2 warp = effect.animation.y * vec2(sin(6.283*p.y+t), cos(6.283*p.x-t));
    vec3 a = texture(albedoTexture, p + warp).rgb;
    vec3 b = texture(albedoTexture, p * 1.7 - warp + vec2(t*0.07, -t*0.05)).rgb;
    vec3 c = texture(albedoTexture, vec2(-p.y,p.x)*0.8 + vec2(-t*0.04,t*0.06)).rgb;
    vec3 sampled = mix(a, (b+c)*0.5, clamp(effect.animation.z,0.0,1.0));
    vec3 pulse = 0.7 + 0.3 * sin(vec3(t,t+2.094,t+4.189));
    return sampled * effect.modulation.rgb * pulse;
#else
    return texture(albedoTexture, uv).rgb;
#endif
}

void main() {
    vec3 N = safeNormalize(worldNormal, vec3(0.0, 1.0, 0.0));
    vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition, N);
    vec3 textureColor = sampleAlbedo(textureUV);
    vec3 baseColor = textureColor * object.albedo.rgb;

    
    vec3 color = baseColor * scene.ambientColorIntensity.rgb *
                 scene.ambientColorIntensity.a;

    vec3 directionalL = safeNormalize(-scene.directionalDirection.xyz, vec3(0,1,0));
    color += blinnPhong(N, V, directionalL,
        scene.directionalColorIntensity.rgb * scene.directionalColorIntensity.a, baseColor);

    
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
#ifdef ENABLE_EFFECTS
    if (effect.control.x != 0u) color += textureColor * effect.modulation.w;
#endif
    outColor = vec4(color, 1.0);
}
