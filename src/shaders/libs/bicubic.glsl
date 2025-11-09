/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef BICUBIC_H
#define BICUBIC_H


// Bicubic sampling from iq: https://www.shadertoy.com/view/XsSXDy
// uses catmull-rom weights

vec4 powers(float x) {return vec4(x*x*x, x*x, x, 1.0);}

vec4 spline(float x, vec4 c0, vec4 c1, vec4 c2, vec4 c3) {
    vec4 ca = vec4( 3.0,  -5.0,   0.0,  2.0 ) / 2.0;
    vec4 cb = vec4(-1.0,   5.0,  -8.0,  4.0 ) / 2.0;

    // We could expand the powers and build a matrix instead (twice as many coefficients
    // would need to be stored, but it could be faster.
    return c0 * dot( cb, powers(x + 1.0)) + 
           c1 * dot( ca, powers(x      )) +
           c2 * dot( ca, powers(1.0 - x)) +
           c3 * dot( cb, powers(2.0 - x));
}

#define SAM(a,b)  texture(tex, (i+vec2(float(a),float(b))+0.5)/res, -99.0)

vec4 texture_Bicubic(sampler2D tex, vec2 t) {
    vec2 res = vec2(textureSize(tex, 0));
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}


#endif // BICUBIC_H