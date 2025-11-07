/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/*
    fxaa_pass.fsh
    -------------
    FXAA pass shader.
*/

#version 460
#extension GL_ARB_shading_language_include: enable

#include "../libs/fxaa.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;

uniform vec2 u_resolution;


void main() {
    vec3 color = fxaa(s_texture, v_uv, u_resolution);

    f_color = vec4(color, 1.0);
}
