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

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform sampler2D s_overlay;

/*
    Bicubic sampling from https://www.shadertoy.com/view/msj3zw
*/

vec4 cubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    vec4 o;
    o.x = s.x;
    o.y = s.y - 4.0 * s.x;
    o.z = s.z - 4.0 * s.y + 6.0 * s.x;
    o.w = 6.0 - o.x - o.y - o.z;
    return o;
}

vec4 textureBicubic(sampler2D tex, vec2 uv){
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 invTexSize = 1.0 / texSize;

    uv = uv * texSize - 0.5;

    vec2 fxy = fract(uv);
    uv -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = uv.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(tex, offset.xz);
    vec4 sample1 = texture(tex, offset.yz);
    vec4 sample2 = texture(tex, offset.xw);
    vec4 sample3 = texture(tex, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
       mix(sample3, sample2, sx),
       mix(sample1, sample0, sx),
       sy
    );
}


void main() {
    vec3 base = textureBicubic(s_texture, v_uv).rgb;
    vec4 overlay = texture(s_overlay, v_uv);
    
    // Source-over alpha blending
    vec3 color = mix(base.rgb, overlay.rgb, overlay.a);

    f_color = vec4(color, 1.0);
}