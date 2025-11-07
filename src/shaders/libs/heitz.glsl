/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef HEITZ_H
#define HEITZ_H


layout(std430, binding = 0) readonly buffer HeitzLayout {
    int heitz_ranking[131072]; // 128 * 128 * 8
    int heitz_scrambling[131072]; // 128 * 128 * 8
    int heitz_sobol[65536]; // 256 * 256
};


struct HeitzState {
    ivec2 pixel;
    int sample_i;
    int dim;
};

HeitzState heitz_state;

/*
    Low-Discrepancy sampler using Owen-scrambled Sobol sequence for Bluenoise

    Algorithm is by Eric Heitz et al.
    https://eheitzresearch.wordpress.com/762-2/

    Parameterization:
    heitz_pixel -> Texture coordinates in screen space
    heitz_dim -> Sample dimension
    heitz_sample_i -> Sample index
*/
float heitz_sample() {
    // TODO: EXPERIMENTAL!, progressive rendering should get better?
    // int HEITZ_CHOOSE_PRNG_AFTER_ACCUMULATED_FRAMES = 16;
    if (heitz_state.dim > 8) {
        return prng();
    }

    int pixel_i = heitz_state.pixel.x;
    int pixel_j = heitz_state.pixel.y;

    // Wrap arguments
    pixel_i = pixel_i & 127;
    pixel_j = pixel_j & 127;
    int sample_idx = heitz_state.sample_i & 255;
    int sample_dim = heitz_state.dim & 255; // TODO: Modulo by 8?

    // XOR index based on optimized ranking
    int ranked_sample_idx = sample_idx ^ heitz_ranking[sample_dim + (pixel_i + pixel_j * 128) * 8];

    if ((sample_dim + (pixel_i + pixel_j * 128) * 8) > (128 * 128 * 8)) {
        issue = true;
    }

    // Fetch value in sequence
    int value = heitz_sobol[sample_dim + ranked_sample_idx * 256];

    if (sample_dim + ranked_sample_idx * 256 > 256 * 256) {
        issue = true;
    }

    // If the dimension is optimized, xor sequence value based on optimized scrambling
    value = value ^ heitz_scrambling[(sample_dim % 8) + (pixel_i + pixel_j * 128) * 8];

    if (((sample_dim % 8) + (pixel_i + pixel_j * 128) * 8) > (128 * 128 * 8)) {
        issue = true;
    }

    // Increase dimension after each call
    heitz_state.dim += 1;

    return (0.5 + float(value)) / 256.0;
}

/*
    Setup Heitz sampling state.

    pixel      -> Integer pixel coordinates in viewport space
    sample_i   -> Sample index
    temporal_i -> Temporal frame index
*/ 
void heitz_seed(ivec2 pixel, int sample_i, int temporal_i) {
    heitz_state.pixel = pixel;
    heitz_state.dim = 0; //(0 + int(u_acc_frame) * 13) % 8; //DIM SET ONCE OUTSIDE SAMPLES?
    // Bluenoise doesn't go well with progressive rendering
    heitz_state.sample_i = sample_i + temporal_i * u_ray_count;
}


#endif // HEITZ_H