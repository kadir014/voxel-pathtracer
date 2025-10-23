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
layout(location = 0) out vec4 f_color;   // COLOR_ATTACHMENT0
layout(location = 1) out vec4 f_normal;  // COLOR_ATTACHMENT1


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
uniform sampler2D s_previous_normal;

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
uniform Camera u_prev_camera;


bool issue;


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


/*
    Low-Discrepancy sampler using Owen-scrambled Sobol sequence for Bluenoise

    Algorithm is by Eric Heitz et al.
    https://eheitzresearch.wordpress.com/762-2/

    Parameterization:
    heitz_pixel -> Texture coordinates in screen space
    heitz_dim -> Sample dimension
    heitz_sample_i -> Sample index
*/
struct HeitzState {
    ivec2 pixel;
    int sample_i;
    int dim;
};
HeitzState heitz_state;
float heitz_sample() {
    // TODO: EXPERIMENTAL!, progressive rendering should get better?
    int HEITZ_CHOOSE_PRNG_AFTER_ACCUMULATED_FRAMES = 16;
    if (u_acc_frame > HEITZ_CHOOSE_PRNG_AFTER_ACCUMULATED_FRAMES) {
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

void heitz_seed(ivec2 pixel, int sample_i, int temporal_i) {
    heitz_state.pixel = pixel;
    heitz_state.dim = 0; //(0 + int(u_acc_frame) * 13) % 8; //DIM SET ONCE OUTSIDE SAMPLES?
    // Bluenoise doesn't go well with progressive rendering
    heitz_state.sample_i = sample_i + temporal_i * u_ray_count;
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
    // vec3 screen_world = u_camera.center + u_camera.u * pos.x + u_camera.v * pos.y;

    // return Ray(
    //     u_camera.position,
    //     normalize(screen_world - u_camera.position)
    // );

    vec3 camera_dir = normalize(u_camera.center - u_camera.position);
	vec3 view_right = u_camera.u;// normalize(cross(vec3(0.0,1.0,0.0),camera_dir));
    vec3 view_up = u_camera.v;//(cross(camera_dir,view_right));
    
    Ray ray;
    ray.origin = u_camera.position;
    ray.dir = normalize(camera_dir * 1.0 + view_right * pos.x + view_up * pos.y);
    
    return ray;
}

vec3 reproject(vec3 world_pos) {
    // z < 0 if invalid

    vec3 to_point = world_pos - u_prev_camera.position;
    vec3 to_point_nrm = normalize(to_point);
    
    vec3 camera_dir = normalize(u_prev_camera.center - u_prev_camera.position);
	vec3 view_right = normalize(u_prev_camera.u);
    vec3 view_up = normalize(u_prev_camera.v);
    
    vec3 fwd = camera_dir;
    
    // too close
    float d = dot(camera_dir,to_point_nrm);
    if(d < 0.01)
        return vec3(0.0, 0.0, -1.0);
    
    d = 1.0 / d;
    
    to_point = to_point_nrm * d - fwd;
    
    float x = dot(to_point,view_right) / length(u_prev_camera.u);
    float y = dot(to_point,view_up) / length(u_prev_camera.v);
    vec2 uv = vec2(x, y);

    // [-1, 1] -> [0, 1]
    uv = uv * 0.50 + 0.50;

    return vec3(uv, 1.0);
}

/*
    Path-trace a single ray and gather radiance information.
*/
vec3 pathtrace(Ray ray, out HitInfo primary_hit) {
    vec3 radiance = vec3(0.0); // Final ray color
    vec3 radiance_delta = vec3(1.0); // Accumulated multiplier

    for (int bounce = 0; bounce < u_bounces; bounce++) {
        
        HitInfo hitinfo = dda(ray);

        if (bounce == 0) {
            primary_hit = hitinfo;
        }

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
    issue = false;

    ivec2 pixel = ivec2(v_uv * u_resolution);

    // Primary hit information in the last sample
    HitInfo primary_hit;

    float u_ray_countf = float(u_ray_count);
    for (int sample_i = 0; sample_i < u_ray_count; sample_i++) {

        // Initialize PRNGs
        int temporal_frame_i = int(u_acc_frame);
        prng_seed(pixel, sample_i, temporal_frame_i);
        heitz_seed(pixel, sample_i, temporal_frame_i);

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

        vec3 radiance = pathtrace(ray, primary_hit);

        final_radiance += radiance / u_ray_countf;
    }

    vec3 final_color = final_radiance;

    f_normal = vec4(0.0);

    if (u_enable_accumulation) {
        vec3 prev_uv = reproject(primary_hit.point);

        if (primary_hit.hit && prev_uv.z > 0.0 && (prev_uv.x > 0.0 && prev_uv.x < 1.0 && prev_uv.y > 0.0 && prev_uv.y < 1.0)) {
            f_normal = vec4(primary_hit.normal, 0.0);

            // Normal xyz  depth a
            vec4 _previous_normal_sample = texture(s_previous_normal, prev_uv.xy);
            vec3 previous_normal = normalize(_previous_normal_sample.rgb);
            float previous_depth = _previous_normal_sample.a;

            float normal_diff = dot(primary_hit.normal, previous_normal);
            if (normal_diff > 0.97) {
                vec3 previous_color = texture(s_previous_frame, prev_uv.xy).rgb;

                // Temporal blending weight
                float weight = 1.0 / float(u_acc_frame + 1u);

                // Blend current and reprojected colors
                final_color = mix(previous_color, final_color, weight);
            }
            else {
                final_color = final_color;
            }
        }
        else {
            // No valid history — reset accumulation
            final_color = final_color;
        }
    }

    f_color = vec4(final_color, luminance(final_color));

    if (issue) {
        f_color = vec4(1000.0, 0.0, 0.0, 1000.0);
    }
}