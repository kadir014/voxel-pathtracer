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

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_texture;
uniform bool u_enable_post;
uniform float u_exposure;
uniform uint u_tonemapper;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;

#define GAMMA 0.45454545454545454545454545454545 // 1.0 / 2.2
#define SQRT2 1.4142135623730950488016887242097 // sqrt(2.0)


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
            Exposure
            --------
            Each value adds one half-stop. 0.0 is neutral exposure.
            - EV = -1.0 -> 0.7x darker
            - EV = 0.0 -> Neutral
            - EV = 1.0 -> 1.4x brighter
        */
        color *= pow(SQRT2, u_exposure);

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
            Human eyes don't see each color equally, we are more sensitive
            to some than others.

            Instead of using 1/3 for each channel, we use a weight distribution
            more suitable for our eyes to determine the luminance of a color.

            Weights are taken from https://en.wikipedia.org/wiki/Relative_luminance

            - A value of 1.0 is neutral and does nothing.
            - Under 1.0 the image approaches grayscale until 0.0.
            - Over 1.0 the colors start to get stronger.
        */
        vec3 y = vec3(0.299, 0.587, 0.114);
        float luminance = dot(color, y);
        color = mix(vec3(luminance), color, u_saturation);

        // Gamma correction, the color is in LDR sRGB space.
        color = pow(color, vec3(GAMMA));
    }

    f_color = vec4(color, 1.0);
}