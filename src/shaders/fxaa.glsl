/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/* OMIT START */
precision mediump float;
/* OMIT END */


/*
    @brief Fast approximate anti-aliasing (FXAA)

    @param tex 2D sampler
    @param uv 2D texture coordinates
    @param resolution Viewport resolution
*/
vec3 fxaa(sampler2D tex, vec2 uv, vec2 resolution) {
    /*
        Edited version of https://www.shadertoy.com/view/4tf3D8

        More on FXAA:
        https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/
    */

	float FXAA_SPAN_MAX = 8.0;
    float FXAA_REDUCE_MUL = 1.0 / 8.0;
    float FXAA_REDUCE_MIN = 1.0 / 128.0;
    vec2 p = uv;

    // 1st stage - Find edges

    // TODO: Optimize division by resolution
    vec3 rgbNW = texture(tex, p + (vec2(-1.0, -1.0) / resolution)).rgb;
    vec3 rgbNE = texture(tex, p + (vec2( 1.0, -1.0) / resolution)).rgb;
    vec3 rgbSW = texture(tex, p + (vec2(-1.0,  1.0) / resolution)).rgb;
    vec3 rgbSE = texture(tex, p + (vec2( 1.0,  1.0) / resolution)).rgb;
    vec3 rgbM  = texture(tex, p).rgb;

    vec3 luma = vec3(0.299, 0.587, 0.114);

    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float lumaSum   = lumaNW + lumaNE + lumaSW + lumaSE;
    float dirReduce = max(lumaSum * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX), dir * rcpDirMin)) / resolution;

    // 2nd stage - Blur

    vec3 rgbA = 0.5 * (
        texture(tex, p + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(tex, p + dir * (2.0 / 3.0 - 0.5)).rgb
    );
    vec3 rgbB =
        rgbA * 0.5 + 0.25 * (
        texture(tex, p + dir * (0.0 / 3.0 - 0.5)).rgb +
        texture(tex, p + dir * (3.0 / 3.0 - 0.5)).rgb
    );

    float lumaB = dot(rgbB, luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    return ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;
}