/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef ATROUS_H
#define ATROUS_H


struct ATrousData {
    sampler2D gi_map;
    sampler2D normal_map;
    sampler2D position_map;

    float gi_phi;
    float normal_phi;
    float position_phi;

    float stepwidth;
    float kernel[25];
    vec2 offset[25];
};

/*
    Edge-Avoiding A-Trous Wavelet Transform

    https://jo.dreggn.org/home/2010_atrous.pdf
*/
vec4 edge_avoiding_atrous(ATrousData data, vec2 p, vec2 resolution) {
    vec4 sum = vec4(0.0);
    vec2 pixel_size = 1.0 / resolution;
    vec4 cval = texture(data.gi_map, p.st);
    vec4 nval = texture(data.normal_map, p.st);
    vec4 pval = texture(data.position_map, p.st);
    float cum_w = 0.0;

    for(int i = 0; i < 25; i++) {
        vec2 uv = p.st + data.offset[i] * pixel_size * data.stepwidth;

        vec4 ctmp = texture(data.gi_map, uv);
        vec4 t = cval - ctmp;
        float dist2 = dot(t, t);
        float c_w = min(exp(-(dist2) / data.gi_phi), 1.0);

        vec4 ntmp = texture(data.normal_map, uv);
        t = nval - ntmp;
        dist2 = max(dot(t, t) / (data.stepwidth * data.stepwidth), 0.0);
        float n_w = min(exp(-(dist2) / data.normal_phi), 1.0);

        vec4 ptmp = texture(data.position_map, uv);
        t = pval - ptmp;
        dist2 = dot(t, t);
        float p_w = min(exp(-(dist2) / data.position_phi), 1.0);

        float weight = c_w * n_w * p_w;
        sum += ctmp * weight * data.kernel[i];
        cum_w += weight * data.kernel[i];
    }

    return sum / cum_w;
}


#endif // ATROUS_H