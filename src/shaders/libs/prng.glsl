/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef PRNG_H
#define PRNG_H


/*
    Wang hash

    From https://www.shadertoy.com/view/ttVGDV
*/
uint wang_hash(uint a) {
    a = (a ^ 61u) ^ (a >> 16);
    a *= 9u;
    a = a ^ (a >> 4);
    a *= 0x27d4eb2du;
    a = a ^ (a >> 15);
    return a;
}


uint prng_state;

/*
    Mulberry32 PRNG
    Returns a float in range 0 and 1.

    From https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
*/
float prng() {
    prng_state += 0x6D2B79F5u;
    uint z = (prng_state ^ (prng_state >> 15)) * (1u | prng_state);
    z ^= z + (z ^ (z >> 7)) * (61u | z);
    return float((z ^ (z >> 14))) / 4294967296.0;
}

/*
    Setup Mulberry32 PRNG seed using Wang hash.

    pixel      -> Integer pixel coordinates in viewport space
    sample_i   -> Sample index
    temporal_i -> Temporal frame index
*/ 
void prng_seed(ivec2 pixel, int sample_i, int temporal_i) {
    // Random big primes
    // XOR the temporal frame number to avoid accumulated patterns
    prng_state = wang_hash(
        uint(pixel.x) * 374761393u +
        uint(pixel.y) * 668265263u +
        (uint(sample_i) * 1597334677u) ^
        (uint(temporal_i) * 3812015801u)
    );
}


#endif // PRNG_H