/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/


#define PI      3.141592653589793238462643383279
#define TAU     6.283185307179586476925286766559
#define INV_PI  0.318309886183790671537767526745 // 1.0 / pi
#define INV_TAU 0.159154943091895335768883763372 // 1.0 / tau

#define GAMMA 0.45454545454545454545454545454545 // 1.0 / 2.2
#define SQRT2 1.4142135623730950488016887242097 // sqrt(2.0)


#define EPSILON 0.0005

// TODO: Do we need to actually store the maximum value?
// single-precision float max = 340282346638528859811704183484516925440.0
#define HIGHP_FLT_MAX 999999.0


#define UPSCALING_METHOD_NEAREST   0
#define UPSCALING_METHOD_BILINEAR  1
#define UPSCALING_METHOD_BICUBIC   2


#define ANTIALIASING_NONE           0
#define ANTIALIASING_JITTERSAMPLING 1
#define ANTIALIASING_FXAA           2


#define NOISE_METHOD_NONE            0
#define NOISE_METHOD_PRNG            1
#define NOISE_METHOD_HEITZ_BLUENOISE 2


/*
    Human eyes don't see each color equally, we are more sensitive
    to some than others.

    Instead of using 1/3 for each channel, we use a weight distribution
    more suitable for our eyes to determine the luminance of a color.

    Weights are taken from https://en.wikipedia.org/wiki/Relative_luminance
*/
vec3 luminance(vec3 color) {
    vec3 y = vec3(0.2125, 0.7154, 0.0721);
    return vec3(dot(color, y));
}


/*
    https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere
*/
vec2 uv_project_sphere(vec3 pos) {
    float u = 0.5 + atan(pos.z, pos.x) / TAU;
    float v = 0.5 + asin(pos.y) / PI;

    return vec2(u, v);
}