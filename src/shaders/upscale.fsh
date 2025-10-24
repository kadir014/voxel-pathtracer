/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/*
    upscale.fsh
    -----------
    Upscaling shader.

    It also overlays the UI texture as a final composite.
*/

#version 460
#extension GL_ARB_shading_language_include: enable

#include "common.glsl"
#include "bicubic.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform sampler2D s_overlay;

uniform vec2 u_resolution;
uniform int u_upscaling_method;


void main() {
    vec3 base = vec3(0.0);
    if (u_upscaling_method == UPSCALING_METHOD_NEAREST) {
        base = texelFetch(s_texture, ivec2(v_uv.x * u_resolution.x, v_uv.y * u_resolution.y), 0).rgb;
    }
    else if (u_upscaling_method == UPSCALING_METHOD_BILINEAR) {
        base = texture(s_texture, v_uv).rgb;
    }
    else if (u_upscaling_method == UPSCALING_METHOD_BICUBIC) {
        base = textureBicubic(s_texture, v_uv).rgb;
    }

    vec4 overlay = texture(s_overlay, v_uv);

    luminance(vec3(1.0));
    
    // Source-over alpha blending
    vec3 color = mix(base.rgb, overlay.rgb, overlay.a);

    f_color = vec4(color, 1.0);
}