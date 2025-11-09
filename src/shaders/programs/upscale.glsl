/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/*
    upscale.fsh
    -----------
    Upscaling shader.

    It also overlays the UI texture as a final composite.
*/

#version 460
#extension GL_ARB_shading_language_include: enable

#include "../libs/common.glsl"
#include "../libs/bicubic.glsl"
#include "../libs/color.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform sampler2D s_overlay;

uniform vec2 u_resolution;
uniform int u_upscaling_method;
uniform int u_target_buffer;


void main() {
    vec4 base = vec4(0.0);
    if (u_upscaling_method == UPSCALING_METHOD_NEAREST) {
        base = texelFetch(s_texture, ivec2(v_uv.x * u_resolution.x, v_uv.y * u_resolution.y), 0);
    }
    else if (u_upscaling_method == UPSCALING_METHOD_BILINEAR) {
        base = texture(s_texture, v_uv);
    }
    else if (u_upscaling_method == UPSCALING_METHOD_BICUBIC) {
        base = texture_Bicubic(s_texture, v_uv);
    }

    vec4 overlay = texture(s_overlay, v_uv);

    if (u_target_buffer == 1) {
        // [-1, 1] -> [0, 1]
        base.rgb = base.rgb * 0.5 + 0.5;
    }
    else if (u_target_buffer == 2) {
        // Depths are stored in normal buffer's alpha
        base.rgb = base.aaa;
    }
    
    // Source-over alpha blending
    vec3 color = mix(base.rgb, overlay.rgb, overlay.a);

    f_color = vec4(color, 1.0);
}