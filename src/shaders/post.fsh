/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/*
    post.fsh
    --------
    HDR color grading and post-processing shader.
*/

#version 460
#extension GL_ARB_shading_language_include: enable
#define INCLUDE_OMIT

#include "common.glsl"

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform bool u_enable_post;
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


/*
    ACES filmic tone mapping curve
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
*/
vec3 aces_filmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0);
}


void main() {
    vec3 color = texture(s_texture, v_uv).rgb;

    if (u_enable_post) {
        /*
            HDR -> LDR sRGB color space pipeline:

            1. The image is in HDR linear space [0, infinity)
            2. Exposure adjustment
            3. Tonemapping
            4. Color grading (contrast, saturation, ...)
            5. Gamma correction and convert to sRGB. [0, 1]
        */

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
            float luma = textureLod(s_texture, v_uv, min_mip).a;

            // Adaptation speed is framerate-dependant, you have to explicitly adjust it
            float target_exposure = log2(0.01 / max(luma, 1e-4)) + u_exposure;
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
        color = mix(luminance(color), color, u_saturation);

        // Gamma correction, the color is in LDR sRGB space.
        color = pow(color, vec3(GAMMA));
    }

    f_color = vec4(color, 1.0);
}