/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef COMMON_H
#define COMMON_H


/******************************************************

                Mathematical Constants

 ******************************************************/

#define PI      3.141592653589793238462643383279
#define TAU     6.283185307179586476925286766559
#define INV_PI  0.318309886183790671537767526745 // 1.0 / pi
#define INV_TAU 0.159154943091895335768883763372 // 1.0 / tau

#define GAMMA   0.454545454545454545454545454545 // 1.0 / 2.2
#define SQRT2   1.414213562373095048801688724209 // sqrt(2.0)

#define EPSILON 0.0005

// TODO: Do we need to actually store the maximum value?
// single-precision float max = 340282346638528859811704183484516925440.0
#define HIGHP_FLT_MAX 999999.0


/******************************************************

                     Enumerations

 ******************************************************/

#define ANTIALIASING_NONE             0
#define ANTIALIASING_JITTERSAMPLING   1
#define ANTIALIASING_FXAA             2

#define NOISE_METHOD_NONE             0
#define NOISE_METHOD_PRNG             1
#define NOISE_METHOD_HEITZ_BLUENOISE  2

#define DENOISER_NONE                 0
#define DENOISER_BILATERAL            1
#define DENOISER_EDGE_AVOIDING_ATROUS 2

#define UPSCALING_METHOD_NEAREST      0
#define UPSCALING_METHOD_BILINEAR     1
#define UPSCALING_METHOD_BICUBIC      2

#define TARGET_BUFFER_PTGI            0
#define TARGET_BUFFER_NORMALS         1
#define TARGET_BUFFER_DEPTH           2
#define TARGET_BUFFER_BOUNCES         3
#define TARGET_BUFFER_POSITONS        4
#define TARGET_BUFFER_ALBEDO          5


/******************************************************

                   Rendering Constants

 ******************************************************/

/*
    No real world material has a reflectance lower than 2% (0.02).
    Use 4% as the default constant for dielectrics for consistency
    with other PBR specifications.
*/
#define DIELECTRIC_BASE_REFLECTANCE 0.04

// GGX can mess up if roughness is exactly 0, so clamp it to a minimum value
#define MIN_ROUGHNESS 0.05

// Index of Refraction of air
#define AIR_IOR 1.0

// Maximum allowed number of steps in DDA
#define MAX_DDA_STEPS 56 // 16 * sqrt(3) * 2

// Factor to multiply emissive surface colors by 
#define EMISSIVE_MULT 2.7


/*
    https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere
*/
vec2 uv_project_sphere(vec3 pos) {
    float u = 0.5 + atan(pos.z, pos.x) / TAU;
    float v = 0.5 + asin(pos.y) / PI;

    return vec2(u, v);
}


#endif // COMMON_H