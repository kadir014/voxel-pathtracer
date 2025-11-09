/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/*
    post.fsh
    --------
    HDR color grading and post-processing shader.
*/

#version 460
#extension GL_ARB_shading_language_include: enable

#include "../libs/common.glsl"
#include "../libs/color.glsl"


in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform sampler2D s_lum;
uniform sampler2D s_albedo;

uniform bool u_enable_post;
uniform float u_chromatic_aberration;
uniform float u_exposure;
uniform uint u_tonemapper;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;

layout(std430, binding = 1) buffer ExposureLayout {
    bool u_enable_eye_adaptation;
    float u_adaptation_speed;
    float u_adapted_exposure;
};


void main() {
    vec3 color = texture(s_texture, v_uv).rgb;

    // Remodulate albedo
    color *= texture(s_albedo, v_uv).rgb;

    if (!u_enable_post) {
        f_color = vec4(color, 1.0);
        return;
    }

    /*
        HDR -> LDR sRGB color space pipeline:

        1. The image is in HDR linear space [0, infinity)
        2. Lens aberration
        3. Exposure adjustment
        4. Tonemapping
        6. Gamma correction and convert to sRGB [0, 1]
        5. Color grading (contrast, saturation, ...)
    */

    /*
        Radial Chromatic Aberration
        ---------------------------
        Imitates lens aberration.

        Since chromatic aberration happens on the lens by affecting the
        individual wavelengths, I think it's most sensible to do this
        in HDR space before any color grading and tonemapping.

        Note: In real life cameras, chromatic aberration mostly affects
        1-2 pixels at max. So keeping the intensity low will be better.
    */
    vec2 delta = v_uv - vec2(0.5);
    float d = length(delta);
    vec2 dir = delta / d;
    
    vec2 nuv = (v_uv - 0.5) * 2.0;
    vec2 offset = dir * (u_chromatic_aberration * d * dot(nuv, nuv));
    
    // Offset outside screen so repeated pixels at the edges don't show
    // Make sure to remodulate new samples with albedo as well
    vec4 g_sample = texture(s_texture, v_uv - offset) * texture(s_albedo, v_uv - offset);
    vec4 b_sample = texture(s_texture, v_uv - offset * 2.0) * texture(s_albedo, v_uv - offset * 2.0);
    color.g = g_sample.g;
    color.b = b_sample.b;

    /*
        Eye adaptation
        --------------
        Adjust the actual exposure applied to the HDR color based on the
        average luminecence of the scene.

        Interpolate between current and target exposure slowly so it mimics
        pupils adjusting to different brightness levels in real life.
    */
    float final_exposure = 0.0;
    if (u_enable_eye_adaptation) {
        // Lowest mipmap level -> log2(max(width, height))
        float min_mip = 10.0;
        float luma = textureLod(s_lum, v_uv, min_mip).a;

        // Adaptation speed is framerate-dependant, you have to explicitly adjust it
        float target_exposure = log2(0.01 / max(luma, EPSILON)) + u_exposure;
        u_adapted_exposure = mix(u_adapted_exposure, target_exposure, u_adaptation_speed);
        u_adapted_exposure = clamp(u_adapted_exposure, -30.0, 10.0);

        final_exposure = u_adapted_exposure;
    }
    else {
        final_exposure = u_exposure;
    }

    /*
        Exposure
        --------
        Each value adds one half-stop. 0.0 is neutral exposure.
        - EV = -1.0 -> 0.7x darker
        - EV = 0.0 -> Neutral
        - EV = 1.0 -> 1.4x brighter
        - EV = 2.0 -> 2.0x brighter (one full-stop)
    */
    color *= pow(SQRT2, final_exposure);

    /*
        Tonemapping
        -----------
        We map the image from HDR to LDR using a tonemap curve.

        In my experience, this often washes out the image,
        so aggressive color grading is needed afterwards.
    */
    if (u_tonemapper == 0) {
        color = clamp(color, 0.0, 1.0);
    }
    else if (u_tonemapper == 1) {
        color = aces_filmic(color);
    }

    /*
        Gamma Correction
        ----------------
        Map the colors onto non-linear sRGB space.
    */
    color = pow(color, vec3(GAMMA));

    /*
        Brightness
        ----------
        - A value of 0.0 is neutral and does nothing.
        - Over or under 0.0, it simply adjusts the overall brightness linearly.
    */
    color += u_brightness;

    /*
        Contrast
        --------
        Pivot around mid-grays and show the difference between darks and lights.

        - A value of 1.0 is neutral and does nothing.
        - Under 1.0, the image loses contrast until 0.0.
        - Over 1.0, the image 'pops up' more.
    */
    color = ((color - 0.5) * max(u_contrast, 0.0)) + 0.5;

    /*
        Saturation
        ----------
        Boost colors depending on human-eye perceived weights.

        - A value of 1.0 is neutral and does nothing.
        - Under 1.0 the image approaches grayscale until 0.0.
        - Over 1.0 the colors start to get stronger.
    */
    color = mix(vec3(luminance(color)), color, u_saturation);

    f_color = vec4(color, 1.0);
} 