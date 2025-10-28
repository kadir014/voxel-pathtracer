/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/* OMIT START */
precision mediump float;
#extension GL_ARB_shading_language_include: enable
/* OMIT END */

#ifndef MICROFACET_H
#define MICROFACET_H

#include "common.glsl"


/*
    No real world material has a reflectance lower than 2% (0.02).
    Use 4% as the default constant for dielectrics for consistency
    with other PBR specifications.
*/
#define DIELECTRIC_BASE_REFLECTANCE 0.04


// Trowbridge-Reitz / GGX Normal Distribution Function
float D_GGX(float NoH, float alpha) {
    float alpha_sqr = alpha * alpha;
    float denom = (NoH * NoH) * (alpha_sqr - 1.0) + 1.0;
    return alpha_sqr / (PI * denom * denom);
}

// Schlick approximation for Freshnel
vec3 F_Schlick(vec3 f0, float VoH) {
    // f90 = 1.0
    return f0 + (vec3(1.0) - f0) * pow(1.0 - VoH, 5.0);
}

// Schlick approximation for GGX geometry term
float G_SchlickGGX(float NdotX, float roughness) {
    /*
        According to UE4 paper, this modification is only for analytical light
        sources and makes the results at glancing angles much darker:
        float r = roughness + 1.0;
        float k = (r * r) / 8.0;

        They also mention choosing k as alpha/2 to match Smith.
    */
    float k = roughness * 0.5;
    return NdotX / (NdotX * (1.0 - k) + k);
}

// g1(v) * g1(l)
float G_Smith(float NdotV, float NdotL, float roughness) {
    float ggx1 = G_SchlickGGX(NdotV, roughness);
    float ggx2 = G_SchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// From UE4 paper
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
vec3 GGX_importance_sample(vec2 xi, float roughness, vec3 N) {
    float alpha = roughness * roughness;
    float phi = TAU * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Spherical coordinates
    vec3 H;
    H.x = sin_theta * cos(phi);
    H.y = sin_theta * sin(phi);
    H.z = cos_theta;

    vec3 up = abs(N.z) < 0.9999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent_x = normalize(cross(up, N));
    vec3 tangent_y = cross(N, tangent_x);

    // Tangent to world space
    return tangent_x * H.x + tangent_y * H.y + N * H.z;
}


#endif // MICROFACET_H