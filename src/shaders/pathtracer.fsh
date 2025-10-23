/*

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

*/

/*
    pathtracer.fsh
    --------------
    Path-traced global illumination shader.
*/

#version 460


//#include src/shaders/common.glsl


#define MAX_DDA_STEPS 56 // 16 * sqrt(3) * 2
#define EMISSIVE_MULT 2.7


in vec2 v_uv;
out vec4 f_color;


uniform int u_ray_count;
uniform int u_bounces;
uniform int u_noise_method;
uniform vec2 u_resolution;
uniform float u_voxel_size;
uniform bool u_enable_roulette;
uniform bool u_enable_sky_texture;
uniform vec3 u_sky_color;
uniform uint u_acc_frame;
uniform bool u_enable_accumulation;
uniform int u_antialiasing;

uniform sampler3D s_grid;
uniform sampler2D s_sky;
uniform sampler2D s_albedo_atlas;
uniform sampler2D s_emissive_atlas;
uniform sampler2D s_roughness_atlas;
uniform sampler2D s_reflectivity_atlas;
uniform sampler2D s_previous_frame;

layout(std430, binding = 0) readonly buffer HeitzLayout {
    int heitz_ranking[131072];
    int heitz_scrambling[131072];
    int heitz_sobol[65536];
};


struct Camera {
    vec3 position;
    vec3 center;
    vec3 u;
    vec3 v;
};

uniform Camera u_camera;


struct Ray {
    vec3 origin; // Origin of ray in world space
    vec3 dir; // Normalized direction of ray
};

struct HitInfo {
    bool hit; // Collision with ray happened?
    vec3 point; // Collision point on the surface in world space
    vec3 normal; // Normal of the collision surface
    int block_id; // ID of the block type
    vec2 face_uv; // Local UV coordinate on the hit face
};


HitInfo dda(Ray ray) {
    HitInfo hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        0,
        vec2(0.0)
    );

    vec3 voxel = ray.origin / u_voxel_size;
    voxel.x = floor(voxel.x);
    voxel.y = floor(voxel.y);
    voxel.z = floor(voxel.z);

    vec3 step_dir = vec3(
        int(ray.dir.x > 0) - int(ray.dir.x < 0),
        int(ray.dir.y > 0) - int(ray.dir.y < 0),
        int(ray.dir.z > 0) - int(ray.dir.z < 0)
    );

    vec3 next_boundary = vec3(
        (voxel.x + (step_dir.x > 0.0 ? 1.0 : 0.0)) * u_voxel_size,
        (voxel.y + (step_dir.y > 0.0 ? 1.0 : 0.0)) * u_voxel_size,
        (voxel.z + (step_dir.z > 0.0 ? 1.0 : 0.0)) * u_voxel_size
    );

    vec3 t_max = next_boundary - ray.origin;
    if (ray.dir.x == 0.0) t_max.x = HIGHP_FLT_MAX;
    else t_max.x /= ray.dir.x;
    if (ray.dir.y == 0.0) t_max.y = HIGHP_FLT_MAX;
    else t_max.y /= ray.dir.y;
    if (ray.dir.z == 0.0) t_max.z = HIGHP_FLT_MAX;
    else t_max.z /= ray.dir.z;

    vec3 t_delta = vec3(0.0);
    if (ray.dir.x == 0.0) t_delta.x = HIGHP_FLT_MAX;
    else t_delta.x = abs(u_voxel_size / ray.dir.x);
    if (ray.dir.y == 0.0) t_delta.y = HIGHP_FLT_MAX;
    else t_delta.y = abs(u_voxel_size / ray.dir.y);
    if (ray.dir.z == 0.0) t_delta.z = HIGHP_FLT_MAX;
    else t_delta.z = abs(u_voxel_size / ray.dir.z);

    // Traverse world texture
    for (int i = 0; i < MAX_DDA_STEPS; i++) {
        // Ray can start out of the map, find a different solution 
        // if (
        //     voxel.x < 0.0 || voxel.y < 0.0 || voxel.z < 0.0 ||
        //     voxel.x >= 16.0 || voxel.y >= 16.0 || voxel.z >= 16.0
        // ) {
        //     if (!started_outside)
        //         return hitinfo;
        // } else if (started_outside) {
        //     started_outside = false;
        // }

        float hit_t = min(t_max.x, min(t_max.y, t_max.z));

        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            voxel.x += step_dir.x;
            t_max.x += t_delta.x;
            hitinfo.normal = vec3(-step_dir.x, 0.0, 0.0);
        }
        else if (t_max.y < t_max.z) {
            voxel.y += step_dir.y;
            t_max.y += t_delta.y;
            hitinfo.normal = vec3(0.0, -step_dir.y, 0.0);
        }
        else {
            voxel.z += step_dir.z;
            t_max.z += t_delta.z;
            hitinfo.normal = vec3(0.0, 0.0, -step_dir.z);
        }

        vec4 voxel_sample = texelFetch(s_grid, ivec3(voxel), 0);
        if (voxel_sample.r > 0.0) {
            hitinfo.hit = true;

            hitinfo.point = ray.origin + ray.dir * hit_t;

            hitinfo.block_id = int(voxel_sample.r * 255.0);

            // WHY DID DIVIDING BY VOXEL SIZE WORK
            vec3 local = fract(hitinfo.point / u_voxel_size);

            // Project onto correct plane
            vec3 abs_n = abs(hitinfo.normal);
            hitinfo.face_uv = local.yz * abs_n.x + local.xz * abs_n.y + local.xy * abs_n.z;

            // TODO: WHAT IS HAPPENING HERE?
            hitinfo.face_uv = mix(hitinfo.face_uv, hitinfo.face_uv.yx, abs_n.x); // swap axes for X faces
            hitinfo.face_uv.x = mix(hitinfo.face_uv.x, 1.0 - hitinfo.face_uv.x, step(0.0, hitinfo.normal.z));
            hitinfo.face_uv.y = mix(hitinfo.face_uv.y, hitinfo.face_uv.y, step(0.0, -hitinfo.normal.y));

            break;
        }
    }

    return hitinfo;
}


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

/*
    Mulberry32 PRNG
    Returns a float in range 0 and 1.

    From https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
*/
uint prng_state;
float prng() {
    prng_state += 0x6D2B79F5u;
    uint z = (prng_state ^ (prng_state >> 15)) * (1u | prng_state);
    z ^= z + (z ^ (z >> 7)) * (61u | z);
    return float((z ^ (z >> 14))) / 4294967296.0;
}


/*
    Bluenoise sampler using Owen-scrambled Sobol sequence

    Algorithm is by Eric Heitz et al.
    https://eheitzresearch.wordpress.com/762-2/

    Parameterization:
    heitz_seed -> Texture coordinates in screen space
    heitz_dim -> Sample dimension
    heitz_sample_i -> Sample index
*/
ivec2 heitz_seed;
int heitz_dim;
int heitz_sample_i;
float heitz_sample() {
    int pixel_i = heitz_seed.x;
    int pixel_j = heitz_seed.y;

    // Wrap arguments
    pixel_i = pixel_i & 127;
    pixel_j = pixel_j & 127;
    int sample_idx = heitz_sample_i & 255;
    int sample_dim = heitz_dim & 255; // TODO: Modulo by 8?

    // XOR index based on optimized ranking
    int ranked_sample_idx = sample_idx ^ heitz_ranking[sample_dim + (pixel_i + pixel_j * 128) * 8];

    // Fetch value in sequence
    int value = heitz_sobol[sample_dim + ranked_sample_idx * 256];

    // If the dimension is optimized, xor sequence value based on optimized scrambling
    value = value ^ heitz_scrambling[(sample_dim % 8) + (pixel_i + pixel_j * 128) * 8];

    // Increase dimension after each call
    heitz_dim++;

    return (0.5 + float(value)) / 256.0;
}

/*
    Generate a random vector in unit sphere.
*/
vec3 random_in_unit_sphere() {
    float r0 = 0.0;
    float r1 = 0.0;
    if (u_noise_method == NOISE_METHOD_PRNG) {
        r0 = prng();
        r1 = prng();
    }
    else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
        r0 = heitz_sample();
        r1 = heitz_sample();
    }

    float z = r0 * 2.0 - 1.0;
    float a = r1 * TAU;
    float r = sqrt(1.0 - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

/*
    Scatter the ray according to the surface material.
*/
Ray brdf(
    Ray ray,
    HitInfo hitinfo,
    inout float specular,
    float roughness,
    float reflectivity
) {
    vec3 new_pos = hitinfo.point + hitinfo.normal * EPSILON;

    float specular_chance = 0.0;
    if (u_noise_method == NOISE_METHOD_PRNG) {
        specular_chance = prng();
    }
    else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
        specular_chance = heitz_sample();
    }

    // TODO: Metallics...

    specular = (specular_chance < reflectivity) ? 1.0 : 0.0;

    vec3 diffuse_ray_dir = normalize(hitinfo.normal + random_in_unit_sphere());
    vec3 specular_ray_dir = reflect(ray.dir, hitinfo.normal);
    specular_ray_dir = normalize(mix(specular_ray_dir, diffuse_ray_dir, roughness * roughness));

    vec3 new_dir = mix(diffuse_ray_dir, specular_ray_dir, specular);

    return Ray(new_pos, new_dir);
}

/*
    Generate ray from camera position to screen position.
*/
Ray generate_ray(vec2 pos) {
    vec3 screen_world = (u_camera.center + u_camera.u * pos.x) + u_camera.v * pos.y;

    return Ray(
        u_camera.position,
        normalize(screen_world - u_camera.position)
    );
}

/*
    Path-trace a single ray and gather radiance information.
*/
vec3 pathtrace(Ray ray) {
    vec3 radiance = vec3(0.0); // Final ray color
    vec3 radiance_delta = vec3(1.0); // Accumulated multiplier

    for (int bounce = 0; bounce < u_bounces; bounce++) {
        
        HitInfo hitinfo = dda(ray);

        // Ray did not hit anything, sample sky
        if (!hitinfo.hit) {
            vec3 sky_color;

            if (u_enable_sky_texture) {
                sky_color = texture(s_sky, uv_project_sphere(ray.dir)).rgb;

                // Sky texture is already tonemapped
                sky_color = pow(sky_color, vec3(2.2));
            }
            else {
                sky_color = u_sky_color;
            }

            radiance += sky_color * radiance_delta;
            break;
        }

        /*
            0 -> top
            1 -> bottom
            2 -> side
        */
        float surface = 0.0;
        if (hitinfo.normal.y > 0.0) surface = 0.0;
        else if (hitinfo.normal.y < 0.0) surface = 1.0;
        else surface = 2.0;

        float atlas_w = 1.0 / 3.0;
        float atlas_h = 1.0 / 7.0; // TODO: Pass block row count as uniform (or pass 1/h)

        vec2 atlas_uv = hitinfo.face_uv;
        atlas_uv.x = atlas_uv.x * atlas_w + float(surface) * atlas_w;
        atlas_uv.y = atlas_uv.y * atlas_h + float(hitinfo.block_id - 1) * atlas_h;

        vec3 albedo = texture(s_albedo_atlas, atlas_uv).rgb;
        vec3 emissive = texture(s_emissive_atlas, atlas_uv).rgb * EMISSIVE_MULT;
        float roughness = texture(s_roughness_atlas, atlas_uv).r;
        float reflectivity = texture(s_reflectivity_atlas, atlas_uv).r;

        float specular = 0.0;
        ray = brdf(ray, hitinfo, specular, roughness, reflectivity);

        vec3 specular_color = vec3(1.0);

        radiance += emissive * radiance_delta;
        radiance_delta *= mix(albedo, specular_color, specular);

        /*
            Russian Roulette:
            As the throughput gets smaller, the ray is more likely to get terminated early.
            Survivors have their value boosted to make up for fewer samples being in the average.
        */
        if (u_enable_roulette) {
            float roulette_result = max(radiance_delta.r, max(radiance_delta.g, radiance_delta.b));

            float roulette_chance = 0.0;
            if (u_noise_method == NOISE_METHOD_PRNG) {
                roulette_chance = prng();
            }
            else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
                roulette_chance = heitz_sample();
            }

            if (roulette_chance > roulette_result) {
                break;
            }
        
            // Add the energy we 'lose' by randomly terminating paths
            radiance_delta *= 1.0 / roulette_result;
        }
    }

    return radiance;
}


void main() {
    vec3 final_radiance = vec3(0.0);

    ivec2 pixel = ivec2(v_uv * u_resolution);

    float u_ray_countf = float(u_ray_count);
    for (int sample_i = 0; sample_i < u_ray_count; sample_i++) {
        prng_state = wang_hash(
            uint(pixel.x) * 1973u +
            uint(pixel.y) * 9277u +
            uint(sample_i) * 26699u +
            uint(u_acc_frame) * 85889u
        );

        heitz_seed = pixel;
        heitz_dim = 0; //DIM SET ONCE OUTSIDE SAMPLES?
        heitz_sample_i = int(u_acc_frame) + sample_i;

        vec2 ray_pos = v_uv * 2.0 - 1.0;

        if (u_antialiasing == ANTIALIASING_JITTERSAMPLING) {
            /*
                Jitter sampling:
                Do antialiasing by sampling the rays with a small amount of jitter.
                This is most effective if progressive rendering is enabled so
                the temporal jitter gets accumulated and averaged.
                But works with single frame renders as well.
                We can use whitenoise PRNG as this doesn't affect UV.
            */
            float pixel_width = 1.0 / u_resolution.x;
            float jitter_amount = pixel_width * 4.0;
            ray_pos += vec2(prng() - 0.5, prng() - 0.5) * jitter_amount;
        }

        Ray ray = generate_ray(ray_pos);

        vec3 radiance = pathtrace(ray);

        final_radiance += radiance / u_ray_countf;
    }

    vec3 final_color = final_radiance;

    if (u_enable_accumulation) {
        vec3 previous_color = texture(s_previous_frame, v_uv).rgb;
        float weight = 1.0 / float(u_acc_frame + uint(1));
        final_color = mix(previous_color, final_color, weight);
    }

    f_color = vec4(final_color, luminance(final_color));
}