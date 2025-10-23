/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/*
    fxaa_pass.fsh
    --------
    FXAA pass shader.
*/

#version 460

//#include src/shaders/fxaa.glsl

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;

uniform vec2 u_resolution;


void main() {
    vec3 color = fxaa(s_texture, v_uv, u_resolution);

    f_color = vec4(color, 1.0);
}