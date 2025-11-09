/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/*
    denoise.fsh
    -----------
    Denoising pass shader.
*/

#version 460
#extension GL_ARB_shading_language_include: enable

#include "../libs/common.glsl"
#include "../libs/bilateral.glsl"
#include "../libs/atrous.glsl"


in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;

uniform int u_denoiser;
uniform vec2 u_resolution;

uniform BilateralData u_bilateral_data;
uniform ATrousData u_atrous_data;


void main() {
    vec3 color = vec3(0.0);

    if (u_denoiser == DENOISER_NONE) {
        color = texture(s_texture, v_uv).rgb;
    }
    else if (u_denoiser == DENOISER_BILATERAL) {
        color = bilateral_filter(s_texture, v_uv, u_bilateral_data, u_resolution);
    }
    else if (u_denoiser == DENOISER_EDGE_AVOIDING_ATROUS) {
        color = edge_avoiding_atrous(u_atrous_data, v_uv, u_resolution).rgb;
    }

    f_color = vec4(color, 1.0);
}