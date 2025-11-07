/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
#extension GL_ARB_shading_language_include: enable
/* OMIT END */

#ifndef BSDF_H
#define BSDF_H

#include "common.glsl"
#include "types.glsl"
#include "color.glsl"
#include "prng.glsl"


/*
    Generate a random vector in unit sphere.
*/
vec3 random_in_unit_sphere() {
    float r0 = 0.0;
    float r1 = 0.0;
    if (u_noise_method == NOISE_METHOD_PRNG) {
        r0 = prng();
        r1 = prng();
    }
    else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
        r0 = heitz_sample();
        r1 = heitz_sample();
    }

    float z = r0 * 2.0 - 1.0;
    float a = r1 * TAU;
    float r = sqrt(1.0 - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}


/*
    Data prepared and packed to be used in scattering functions.

    albedo
        Albedo color processed from surface material
    metallic
        Metallic value processed from surface material
    roughness
        Linear (not perceptual) roughness value processed from surface material
    f0
        F0 base reflectance value
    ior
        Index of refraction value processed from surface material
    diffuse_weight
        Normalized weight of the diffuse lobe
    specular_weight
        Normalized weight of the specular lobe
    transmit_weight
        Normalized weight of the transmission lobe
    lobe
        Randomly chosen lobe event value
*/
struct BSDFState {
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 f0;
    float ior;
    float diffuse_weight;
    float specular_weight;
    float transmit_weight;
    float lobe;
};

/*
    Prepare material properties and lobe weights for sampling the BSDF.
*/
BSDFState prepare_bsdf(Material material) {
    // Albedo over 1.0 can result in emissive-like behavior, but I'm not sure if this is accurate
    vec3 albedo = clamp(material.albedo, 0.0, 1.0);
    float metallic = clamp(material.metallic, 0.0, 1.0);
    float transmission = clamp(material.glass, 0.0, 1.0);
    float ior = max(material.ior, 1.0);

    // Perceptual roughness -> linear material roughness
    // Often called alpha in BRDF equations
    float perceived_roughness = clamp(material.roughness, MIN_ROUGHNESS, 1.0);
    float roughness = perceived_roughness * perceived_roughness;

    float reflectance = clamp(material.reflectance, DIELECTRIC_BASE_REFLECTANCE, 1.0);

    // Use albedo to tint metals
    vec3 f0 = mix(vec3(reflectance), albedo, metallic);

    // Diffuse reflectance weight
    float diffuse_weight = (1.0 - metallic) * (1.0 - transmission);

    // Specular reflectance weight
    // Use average Fresnel at normal incidence (perceived weights)
    float specular_weight = luminance(f0);

    // Specular refraction weight
    float transmit_weight = transmission * (1.0 - metallic);

    // Normalize weights so they sum up to 1.0
    float inv_weight = 1.0 / (diffuse_weight + specular_weight + transmit_weight);
    diffuse_weight *= inv_weight;
    specular_weight *= inv_weight;
    transmit_weight *= inv_weight;

    float lobe = 0.0;
    if (u_noise_method == NOISE_METHOD_PRNG) {
        lobe = prng();
    }
    else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
        lobe = heitz_sample();
    }

    return BSDFState(
        albedo,
        metallic,
        roughness,
        f0,
        ior,
        diffuse_weight,
        specular_weight,
        transmit_weight,
        lobe
    );
}

/*
    Diffuse BRDF (Bidirectional Reflectance Distribution Function)
    
    Lambertian model:
    -----------------
    f = albedo / pi
    pdf = NoL / pi

    Parameters
    ----------
    V        -> View direction (incoming ray)
    N        -> Surface normal
    L        -> Reflected ray direction
    state    -> BSDF state
    pdf      -> PDF of the sampled BRDF

    return   -> Sampled radiance (multiplied by NoL)
*/
vec3 diffuse_brdf(
    vec3 V,
    vec3 N,
    vec3 L,
    BSDFState state,
    out float pdf
) {
    float NoL = dot(N, L);
    if (NoL <= 0.0) {
        pdf = 0.0;
        return vec3(0.0);
    }

    vec3 brdf = state.albedo / PI;
    brdf *= NoL;

    pdf = NoL / PI;
    pdf *= state.diffuse_weight;

    return brdf;
}

/*
    Specular BRDF (Bidirectional Reflectance Distribution Function)

    Cook-Torrance microfacet model:
    -------------------------------
    D -> Normal distrubiton function term (How microfacets are oriented)
    F -> Fresnel term
    G -> Geometry term (How many microfacets are visible)

    f = D * G * F / (4 * NoL * NoV)
    pdf = D * NoH / (4 * VoH)

    Parameters
    ----------
    V        -> View direction (incoming ray)
    N        -> Surface normal
    L        -> Reflected ray direction
    H        -> Half vector
    state    -> BSDF state
    pdf      -> PDF of the sampled BRDF

    return   -> Sampled radiance (multiplied by NoL)
*/
vec3 specular_brdf(
    vec3 V,
    vec3 N,
    vec3 L,
    vec3 H,
    BSDFState state,
    out float pdf
) {
    float NoL = dot(N, L);
    float NoV = dot(N, V);

    if (NoL <= 0.0 || NoV <= 0.0) {
        pdf = 0.0;
        return vec3(0.0);
    }

    float NoH = dot(N, H);
    float VoH = dot(V, H);

    // TODO: abs instead of clamp?
    NoL = clamp(NoL, 0.0, 1.0);
    NoV = clamp(NoV, 0.0, 1.0);
    NoH = clamp(NoH, 0.0, 1.0);
    VoH = clamp(VoH, 0.0, 1.0);

    float D = D_GGX(NoH, state.roughness);
    float G = G_Smith(NoV, NoL, state.roughness);
    vec3 F = F_Schlick(state.f0, VoH);

    vec3 brdf = (D * G * F) / (4.0 * NoL * NoV);
    brdf *= NoL;

    pdf = D * NoH / (4.0 * VoH);
    pdf *= state.specular_weight;

    return brdf;
}

/*
    Specular BTDF (Bidirectional Transmittance Distribution Function)

    B. Walter et al. Rough refraction microfacet model:
    ---------------------------------------------------
    D -> Normal distribution function
    F -> Fresnel term
    G -> Shadowing-masking function
    ni -> Incident IOR
    no -> Material IOR

    f = ((VoH * LoH) / (NoV * NoL)) * ((no^2 * (1-F) * G * D) / (ni*VoH + no*LoH)^2)
    pdf = ???

    Parameters
    ----------
    V        -> View direction (incoming ray)
    N        -> Surface normal
    L        -> Reflected ray direction
    H        -> Half vector
    inside   -> Started inside the medium or not?
    state    -> BSDF state
    pdf      -> PDF of the sampled BTDF

    return   -> Sampled radiance (multiplied by NoL)
*/
vec3 specular_btdf(
    vec3 V,
    vec3 N,
    vec3 L,
    vec3 H,
    bool inside,
    BSDFState state,
    out float pdf
) {

    float ni = AIR_IOR;
    float no = state.ior;

    if (inside) {
        ni = state.ior;
        no = AIR_IOR;
    }

    float eta = ni / no;

    // ? abs because we're in the lower lobe????
    float NoV = abs(dot(N, V));
    float NoL = abs(dot(N, L));
    float NoH = abs(dot(N, H));
    float VoH  = abs(dot(V, H));
    float LoH  = abs(dot(L, H));

    // TODO: Do tests on which ones to clamp and test for NaNs

    // if (NoV <= 0.0 || NoL <= 0.0) {
    //     pdf = 0.0;
    //     return vec3(0.0);
    // }

    // NoL = clamp(NoL, 0.0, 1.0);
    // NoV = clamp(NoV, 0.0, 1.0);
    // NoH = clamp(NoH, 0.0, 1.0);
    // VoH = clamp(VoH, 0.0, 1.0);
    // LoH = clamp(LoH, 0.0, 1.0);

    // NoH = min(0.99, NoH);

    float D = D_GGX(NoH, state.roughness);
    float G = G_Smith(NoV, NoL, state.roughness);
    vec3 F = F_Schlick(state.f0, VoH);

    // (ni*VoH + no*LoH)^2 part of the BTDF
    float denom = (ni * VoH + no * LoH);
    denom = denom * denom;

    //vec3 btdf = ((VoH * LoH) / (NoV * NoL)) * ((no*no * (1.0 - F) * G * D) / denom);

    //vec3 btdf = abs(VoH * LoH) * D * G * (vec3(1.0) - F);
    //btdf *= (no * no) / max(denom, 1e-8);
    //btdf /= max(NoV * NoL, 1e-8); // /(|n⋅v| |n⋅l|)

    float jacobian = abs(LoH) / denom;

    vec3 abso = pow(state.albedo, vec3(0.5));
    vec3 btdf = abso * D * G * (1.0 - F) * abs(VoH) * jacobian * pow(eta, 2.0) / abs(NoL * NoV);

    // ? pdf = D * NoH * abs(LoH) / denom;

    pdf = G1_SchlickGGX(abs(NoL), state.roughness) * max(0.0, VoH) * D * jacobian / NoV;

    return btdf * NoL;
}

/*
    My physically-based BSDF (Bidirectional Scattering Distribution Function)

    BRDF is based on UE4's model in Brian Karis' paper.
    BTDF is based on B. Walter et al. paper.

    Parameters
    ----------
    V        -> View direction (incoming ray)
    N        -> Surface normal
    inside   -> Started inside the medium or not?
    material -> Surface material
    L        -> Reflected ray direction
    pdf      -> PDF of the sampled BSDF

    return   -> Sampled radiance (multiplied by NoL)
*/
vec3 sample_bsdf(
    vec3 V,
    inout vec3 N,
    bool inside,
    Material material,
    out vec3 L,
    out float pdf
) {
    BSDFState state = prepare_bsdf(material);

    L = vec3(0.0);
    pdf = 0.0;

    // Sample diffuse BRDF
    if (state.lobe < state.diffuse_weight) {
        // Cosine weighted hemisphere
        L = normalize(N + random_in_unit_sphere());

        return diffuse_brdf(V, N, L, state, pdf);
    }

    // Specular event (reflection or refraction)
    else {
        vec2 xi = vec2(0.0);
        if (u_noise_method == NOISE_METHOD_PRNG) {
            xi.x = prng();
            xi.y = prng();
        }
        else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
            xi.x = heitz_sample();
            xi.y = heitz_sample();
        }

        vec3 H = GGX_importance_sample(xi, state.roughness, N);
        float VoH = dot(V, H);

        // Sample specular BRDF
        if (state.lobe < state.diffuse_weight + state.specular_weight) {
            L = 2.0 * VoH * H - V;

            return specular_brdf(V, N, L, H, state, pdf);
        }

        // Sample specular BTDF
        else if (state.transmit_weight > 0.0) {
            float eta = inside ? state.ior / AIR_IOR : AIR_IOR / state.ior;

            // TODO: Handle total internal reflection (L=vec3(0))
            L = refract(-V, H, eta);

            if (L == vec3(0.0)) {
                pdf = 0.0;
                return vec3(0.0);
            }

            vec3 btdf = specular_btdf(V, N, L, H, inside, state, pdf);

            // Use the inverse lobe when spawning the next ray
            N = -N;

            return btdf;
        }
    }

    return vec3(0.0);
}


#endif // BSDF_H