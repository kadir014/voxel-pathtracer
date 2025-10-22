/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/


#define PI      3.141592653589793238462643383279
#define TAU     6.283185307179586476925286766559
#define INV_PI  0.318309886183790671537767526745
#define INV_TAU 0.159154943091895335768883763372


#define EPSILON 0.0005

// TODO: Do we need to actually store the maximum value?
// single-precision float max = 340282346638528859811704183484516925440.0
#define HIGHP_FLT_MAX 999999.0


#define ANTIALIASING_NONE           0
#define ANTIALIASING_JITTERSAMPLING 1
#define ANTIALIASING_FXAA           2