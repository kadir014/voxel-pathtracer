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


#define PI 3.141592653589793238462643383279
#define TAU 6.283185307179586476925286766559
#define EPSILON 0.0005
#define BIG_VALUE 10000.0
#define BLUENOISE_SIZE 1024
#define VOXEL_SIZE 5.0
#define MAX_DDA_STEPS 34
#define GRID_SIZE 10


in vec2 v_uv;
out vec4 f_color;


uniform int u_ray_count;
uniform int u_bounces;
uniform int u_noise_method;
uniform vec2 u_resolution;

uniform sampler2D s_bluenoise;
uniform sampler3D s_grid;
uniform sampler2D s_sky;
uniform sampler2D s_albedo_atlas;
uniform sampler2D s_emissive_atlas;


struct Camera {
    vec3 position;
    vec3 center;
    vec3 u;
    vec3 v;
};

uniform Camera u_camera;


struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Material {
    vec3 color;
    vec3 emissive;
    float specular_percentage;
    vec3 specular_color;
    float roughness;
};

struct HitInfo {
    bool hit; // Collision with ray happened?
    vec3 point; // Collision point on the surface in world space
    vec3 normal; // Normal of the collision surface
    int block_id;
    vec2 face_uv;
    //Material material;
};


bool out_of_grid(vec3 voxel) {
    float size = float(GRID_SIZE);

    return (voxel.x < 0.0 || voxel.y < 0.0 || voxel.z < 0.0 ||
            voxel.x >= size || voxel.y >= size || voxel.z >= size);
}

bool is_solid(vec3 voxel) {
    vec4 s = texelFetch(s_grid, ivec3(voxel), 0);

    if (s.r > 0.0) {
        return true;
    }

    return false;
}

vec3 sign_vec3(vec3 v) {
    return vec3(
        int(v.x > 0) - int(v.x < 0),
        int(v.y > 0) - int(v.y < 0),
        int(v.z > 0) - int(v.z < 0)
    );
}

HitInfo dda(Ray ray) {
    HitInfo hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        0,
        vec2(0.0)
        //Material(vec3(1.0, 1.0, 1.0), vec3(0.0), 1.0, vec3(1.0), 0.2)
    );

    vec3 voxel = ray.origin / VOXEL_SIZE;
    voxel.x = floor(voxel.x);
    voxel.y = floor(voxel.y);
    voxel.z = floor(voxel.z);

    vec3 step_dir = sign_vec3(ray.dir);

    vec3 next_boundary = vec3(
        (voxel.x + (step_dir.x > 0.0 ? 1.0 : 0.0)) * VOXEL_SIZE,
        (voxel.y + (step_dir.y > 0.0 ? 1.0 : 0.0)) * VOXEL_SIZE,
        (voxel.z + (step_dir.z > 0.0 ? 1.0 : 0.0)) * VOXEL_SIZE
    );

    vec3 t_max = next_boundary - ray.origin;
    if (ray.dir.x == 0.0) t_max.x = BIG_VALUE;
    else t_max.x /= ray.dir.x;
    if (ray.dir.y == 0.0) t_max.y = BIG_VALUE;
    else t_max.y /= ray.dir.y;
    if (ray.dir.z == 0.0) t_max.z = BIG_VALUE;
    else t_max.z /= ray.dir.z;

    vec3 t_delta = vec3(0.0);
    if (ray.dir.x == 0.0) t_delta.x = BIG_VALUE;
    else t_delta.x = abs(VOXEL_SIZE / ray.dir.x);
    if (ray.dir.y == 0.0) t_delta.y = BIG_VALUE;
    else t_delta.y = abs(VOXEL_SIZE / ray.dir.y);
    if (ray.dir.z == 0.0) t_delta.z = BIG_VALUE;
    else t_delta.z = abs(VOXEL_SIZE / ray.dir.z);

    // Traverse
    for (int i = 0; i < MAX_DDA_STEPS; i++) {
        // ray can start out of the map
        // if (out_of_grid(voxel)) {
        //     return false;
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

            // if (voxel.y > 0.0) {
            //     hitinfo.material.emissive = vec3(1.0) * 1.5;
            // }

            hitinfo.point = ray.origin + ray.dir * hit_t;

            hitinfo.block_id = int(voxel_sample.r * 255.0);

            // WHY DID DIVIDING BY VOXEL SIZE WORK
            vec3 local = fract(hitinfo.point / VOXEL_SIZE);

            // Project onto correct plane (branchless)
            vec3 abs_n = abs(hitinfo.normal);
            hitinfo.face_uv = local.yz * abs_n.x + local.xz * abs_n.y + local.xy * abs_n.z;

            // TODO: WTF IS HAPPENING HERE?
            vec2 uv = hitinfo.face_uv;
            uv = mix(uv, uv.yx, abs_n.x); // swap axes for X faces
            uv.x = mix(uv.x, 1.0 - uv.x, step(0.0, hitinfo.normal.z));
            uv.y = mix(uv.y, uv.y, step(0.0, -hitinfo.normal.y));

            hitinfo.face_uv = uv;

            break;
        }
    }

    return hitinfo;
}


/*
    https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere
*/
vec2 uv_project_sphere(vec3 pos) {
    float u = 0.5 + atan(pos.z, pos.x) / TAU;
    float v = 0.5 + asin(pos.y) / PI;

    return vec2(u, v);
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

vec2 bluenoise_seed;
vec4 bluenoise() {
    vec4 bluenoise_sample = texture(s_bluenoise, (bluenoise_seed * u_resolution) / vec2(BLUENOISE_SIZE, BLUENOISE_SIZE));
    return fract(bluenoise_sample);
}


vec3 random_in_unit_sphere() {
    float r0 = 0.0;
    float r1 = 0.0;
    if (u_noise_method == 1) {
        r0 = prng();
        r1 = prng();
    }
    else if (u_noise_method == 2) {
        r0 = bluenoise().g;
        r1 = bluenoise().b;
    }

    float z = r0 * 2.0 - 1.0;
    float a = r1 * TAU;
    float r = sqrt(1.0 - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

Ray scatter(Ray ray, HitInfo hitinfo, inout float specular) {
    vec3 new_pos = hitinfo.point + hitinfo.normal * EPSILON;

    float specular_chance = 0.0;
    if (u_noise_method == 1) {
        specular_chance = prng();
    }
    else if (u_noise_method == 2) {
        specular_chance = bluenoise().r;
    }

    float specular_percentage = 0.0;
    float roughness = 0.0;

    specular = (specular_chance < specular_percentage) ? 1.0 : 0.0;

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
            vec3 sky_color = texture(s_sky, uv_project_sphere(ray.dir)).rgb;

            // Sky texture is already tonemapped
            sky_color = pow(sky_color, vec3(2.2));

            sky_color *= 0.3;

            radiance += sky_color * radiance_delta;
            break;
        }

        float specular = 0.0;
        ray = scatter(ray, hitinfo, specular);

        vec3 specular_color = vec3(1.0);

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
        float atlas_h = 1.0 / 4.0;

        vec2 atlas_uv = hitinfo.face_uv;
        atlas_uv.x = atlas_uv.x * atlas_w + float(surface) * atlas_w;
        atlas_uv.y = atlas_uv.y * atlas_h + float(hitinfo.block_id - 1) * atlas_h;

        vec3 albedo = texture(s_albedo_atlas, atlas_uv).rgb;
        vec3 emissive = texture(s_emissive_atlas, atlas_uv).rgb * 3.0;

        radiance += emissive * radiance_delta;
        radiance_delta *= mix(albedo, specular_color, specular);

        // /*
        //     Russian Roulette:
        //     As the throughput gets smaller, the ray is more likely to get terminated early.
        //     Survivors have their value boosted to make up for fewer samples being in the average.
        // */
        // if (u_roulette) {
        //     float roulette_result = max(ray_color.r, max(ray_color.g, ray_color.b));
        //     if (prng(prng_state) > roulette_result) {
        //         break;
        //     }
        
        //     // Add the energy we 'lose' by randomly terminating paths
        //     ray_color *= 1.0 / roulette_result;
        // }
    }

    return radiance;
}


void main() {
    vec3 final_radiance = vec3(0.0);

    vec2 pixel = v_uv * u_resolution;

    bluenoise_seed = v_uv;

    float u_ray_countf = float(u_ray_count);
    for (int sample_i = 0; sample_i < u_ray_count; sample_i++) {
        prng_state = wang_hash(
            uint(pixel.x) * 1973u +
            uint(pixel.y) * 9277u +
            uint(sample_i) * 26699u
        );

        Ray ray = generate_ray(v_uv * 2.0 - 1.0);

        vec3 radiance = pathtrace(ray);

        final_radiance += radiance / u_ray_countf;
    }

    // if (v_uv.x > 0.5) {
    //     final_radiance = texture(s_emissive_atlas, vec2((v_uv.x + 0.5) * 2.0, v_uv.y)).rgb;
    // }

    f_color = vec4(final_radiance, 1.0);
}