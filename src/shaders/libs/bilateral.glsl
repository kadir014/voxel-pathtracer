/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef BILATERAL_H
#define BILATERAL_H


struct BilateralData {
    int hw;
    float sigmaspace;
    float sigmacolor;
};

vec3 bilateral_filter(
    sampler2D tex,
    vec2 uv,
    BilateralData data,
    vec2 resolution
) {
    // Slightly edited version of https://www.shadertoy.com/view/NlKczy

    vec4 center = texture(tex, uv);

    float Ss = pow(data.sigmaspace, 2.0) * 2.0;
    float Sc = pow(data.sigmacolor, 2.0) * 2.0;

    vec4 TW = vec4(0.0); // Sum of weights
    vec4 WI = vec4(0.0); // Sum of weighted intensities
    vec4 w = vec4(0.0);

    for (int i = -data.hw; i <= data.hw; i++) {
        for (int j = -data.hw; j <= data.hw; j++) {
            vec2 dx = vec2(float(i), float(j));
            vec2 tc = uv + dx / resolution;
            vec4 Iw = texture(tex, tc);
            vec4 dc = (center - Iw) * 255.0;

            w = exp(-dot(dx, dx) / Ss - dc*dc / Sc);
            TW += w;
            WI += Iw * w;
        }
    }
    
    return (WI / TW).rgb;
}


#endif // BILATERAL_H