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

#include "common.glsl"


in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;

uniform int u_denoiser;
uniform vec2 u_resolution;

uniform int u_hw;
uniform float u_sigmaspace;
uniform float u_sigmacolor;


vec3 bilateral_filter(sampler2D tex, vec2 uv) {
    // https://www.shadertoy.com/view/NlKczy

    vec4 center = texture(tex, uv);

    float Ss = pow(u_sigmaspace, 2.0) * 2.0;
    float Sc = pow(u_sigmacolor, 2.0) * 2.0;

    vec4 TW = vec4(0.0); // Sum of Weights
    vec4 WI = vec4(0.0); // Sum of Weighted Intensities
    vec4 w = vec4(0.0);

    for (int i = -u_hw; i <= u_hw; i++) {
        for (int j = -u_hw; j <= u_hw; j++) {
            vec2 dx = vec2(float(i), float(j));
            vec2 tc = uv + dx / u_resolution;
            vec4 Iw = texture(tex, tc);
            vec4 dc = (center - Iw) * 255.0;

            w = exp(-dot(dx, dx) / Ss - dc*dc / Sc);
            TW += w;
            WI += Iw * w;
        }
    }
    
    return (WI / TW).rgb;
}


void main() {
    vec3 color = vec3(0.0);

    if (u_denoiser == DENOISER_NONE) {
        color = texture(s_texture, v_uv).rgb;
    }
    else if (u_denoiser == DENOISER_BILATERAL) {
        color = bilateral_filter(s_texture, v_uv);
    }

    f_color = vec4(color, 1.0);
}
