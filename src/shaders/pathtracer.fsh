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
#extension GL_ARB_shading_language_include: enable

#include "common.glsl"
#include "microfacet.glsl"


#define MAX_DDA_STEPS 56 // 16 * sqrt(3) * 2
#define EMISSIVE_MULT 2.7


in vec2 v_uv;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_bounces;


uniform int u_ray_count;
uniform int u_bounces;
uniform int u_noise_method;
uniform vec2 u_resolution;
uniform float u_voxel_size;
uniform bool u_enable_roulette;
uniform bool u_enable_sky_texture;
uniform bool u_enable_nee;
uniform vec3 u_sky_color;
uniform bool u_enable_accumulation;
uniform int u_antialiasing;
uniform int u_exp_raymarch;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_radiance;
uniform float u_sun_angular_radius;

uniform sampler3D s_grid;
uniform sampler2D s_sky;
uniform sampler2D s_albedo_atlas;
uniform sampler2D s_emissive_atlas;
uniform sampler2D s_metallic_atlas;
uniform sampler2D s_roughness_atlas;
uniform sampler2D s_previous_frame;
uniform sampler2D s_previous_normal;

layout(std430, binding = 0) readonly buffer HeitzLayout {
    int heitz_ranking[131072];
    int heitz_scrambling[131072];
    int heitz_sobol[65536];
};

layout(std430, binding = 2) buffer AccLayout {
    int accumulations[]; // Logical width x Logical height
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

// Ray x Voxel hit information
struct HitInfo {
    bool hit; // Collision with ray happened?
    vec3 point; // Collision point on the surface in world space
    vec3 normal; // Normal of the collision surface
    int block_id; // ID of the block type
    vec2 face_uv; // Local UV coordinate on the hit face
};

// Physically-based surface material
struct Material {
    vec3 albedo; // Base color (should have minimal or no shadows) [0, 1] 
    vec3 emissive; // Emission color [0, infinity]
    float metallic; // 0 = Dielectric 1 = Metallic
    float roughness; // Perceptual roughness [0, 1]
    float reflectance; // Base reflectance (F0) [0, 1] 
};

// Material preview feature only, will be removed
uniform Material u_exp_material;


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


// SDFs are from https://iquilezles.org/articles/distfunctions/

float sdf_sphere(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float sdf_box(vec3 p, vec3 center, vec3 b) {
    vec3 q = abs((p - center)) - b;
    return length(max(q, 0.0)) + min(max(q.x,max(q.y,q.z)), 0.0);
}

float sdf_round_box(vec3 p, vec3 center, vec3 b, float r) {
    vec3 q = abs((p - center)) - b + r;
    return length(max(q, 0.0)) + min(max(q.x,max(q.y,q.z)), 0.0) - r;
}

float sdf(vec3 pos) {
   //return min(sdf_sphere(pos, vec3(0.0), 1.5), sdf_box(pos, vec3(0.0), vec3(1.0)));

   return sdf_sphere(pos, vec3(0.0), 1.0);
   //return sdf_round_box(pos, vec3(0.0), vec3(1.0), 0.5);
   //return sdf_box(pos, vec3(0.0), vec3(1.0));
}

vec3 estimate_normal(vec3 p) {
    return normalize(vec3(
        sdf(vec3(p.x + EPSILON, p.y, p.z)) - sdf(vec3(p.x - EPSILON, p.y, p.z)),
        sdf(vec3(p.x, p.y + EPSILON, p.z)) - sdf(vec3(p.x, p.y - EPSILON, p.z)),
        sdf(vec3(p.x, p.y, p.z  + EPSILON)) - sdf(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

HitInfo raymarch(Ray ray) {
    HitInfo hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        0,
        vec2(0.0)
    );

    vec3 curr_pos = ray.origin;

    for (int i = 0; i < 100; i++) {
    	float dist = sdf(curr_pos);

        if (dist < EPSILON) {
            return HitInfo(
                true,
                curr_pos,
                estimate_normal(curr_pos),
                0,
                vec2(0.0, 0.0)
            );
        }

        curr_pos += ray.dir * dist;
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


struct BRDFState {
    vec3 albedo;
    float metallic;
    float roughness;
    vec3 f0;
    float diffuse_weight;
    float specular_weight;
    float lobe;
};

/*
    Prepare material properties and lobe weights for sampling BRDFs.
*/
BRDFState prepare_brdf(Material material) {
    vec3 albedo = material.albedo;
    float metallic = clamp(material.metallic, 0.0, 1.0);

    // Perceptual roughness -> linear material roughness
    // Often called alpha in BRDF equations
    // Roughness of 0 can mess up with D term
    float min_roughness = 0.05;
    float perceived_roughness = clamp(material.roughness, min_roughness, 1.0);
    float roughness = perceived_roughness * perceived_roughness;

    float reflectance = clamp(material.reflectance, DIELECTRIC_BASE_REFLECTANCE, 1.0);

    // Use albedo to tint metals
    vec3 f0 = mix(vec3(reflectance), albedo, metallic);

    // Diffuse reflectance weight
    float diffuse_weight = 1.0 - metallic;

    // Specular reflectance weight
    // Use average Fresnel at normal incidence (perceived weights)
    float specular_weight = luminance(f0);

    // Adjust weights so they sum up to 1.0
    float inv_weight = 1.0 / (diffuse_weight + specular_weight);
    diffuse_weight *= inv_weight;
    specular_weight *= inv_weight;

    float lobe = 0.0;
    if (u_noise_method == NOISE_METHOD_PRNG) {
        lobe = prng();
    }
    else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
        lobe = heitz_sample();
    }

    return BRDFState(
        albedo,
        metallic,
        roughness,
        f0,
        diffuse_weight,
        specular_weight,
        lobe
    );
}

vec3 diffuse_brdf(
    vec3 V,
    vec3 N,
    vec3 L,
    BRDFState state,
    out float pdf
) {
    /*
        Sample diffuse BRDF

        Lambertian model:
        -----------------
        f = albedo / pi
        pdf = NoL / pi
    */

    float NoL = dot(N, L);
    if (NoL <= 0.0) {
        pdf = 0.0;
        return vec3(0.0);
    }

    vec3 brdf = state.albedo / PI;
    brdf *= NoL;

    pdf = NoL / PI;
    pdf *= state.diffuse_weight;

    return brdf;
}

vec3 specular_brdf(
    vec3 V,
    vec3 N,
    vec3 L,
    vec3 H,
    BRDFState state,
    out float pdf
) {
    /*
        Sample specular BRDF

        Cook-Torrance microfacet model:
        -------------------------------
        D -> Normal distrubiton function term (How microfacets are oriented)
        F -> Fresnel term
        G -> Geometry term (How many microfacets are visible)

        f = D * G * F / (4 * NoL * NoV)
        pdf = D * NoH / (4 * VoH)
    */

    float NoL = dot(N, L);
    float NoV = dot(N, V);

    if (NoL <= 0.0 || NoV <= 0.0) {
        pdf = 0.0;
        return vec3(0.0);
    }

    float NoH = dot(N, H);
    float VoH = dot(V, H);

    NoL = clamp(NoL, 0.0, 1.0);
    NoV = clamp(NoV, 0.0, 1.0);
    NoH = clamp(NoH, 0.0, 1.0);
    VoH = clamp(VoH, 0.0, 1.0);

    float D = D_GGX(NoH, state.roughness);
    float G = G_Smith(NoV, NoL, state.roughness);
    vec3 F = F_Schlick(state.f0, VoH);

    vec3 brdf = (D * G * F) / (4.0 * NoL * NoV);
    brdf *= NoL;

    pdf = D * NoH / (4.0 * VoH);
    pdf *= state.specular_weight;

    return brdf;
}

/*
    My physically-based material BRDF.
    Heavily inspired by UE4's sampler showcased in Brian Karis' paper.

    Uses a "metalness" workflow.
    Lambertian BRDF for diffuse and Cook-Torrance microfacet model for specular.

    N        -> Surface normal
    V        -> View direction
    material -> Surface material
    L        -> Reflected ray direction
    pdf      -> PDF of the sampled BRDF
    return   -> Sampled radiance (multiplied by NoL)
*/
vec3 sample_brdf(
    vec3 N,
    vec3 V,
    Material material,
    out vec3 L,
    out float pdf
) {
    /*
        1. Choose which reflectance to sample (diffuse or specular)
        2. Calculate L (reflected ray direction)
        3. Sample weighted BRDFs
    */

    BRDFState state = prepare_brdf(material);

    // Sample diffuse BRDF
    if (state.lobe < state.diffuse_weight) {
        // Cosine weighted hemisphere
        L = normalize(N + random_in_unit_sphere());

        return diffuse_brdf(V, N, L, state, pdf);
    }
    // Sample specular BRDF
    else {
        vec2 xi = vec2(0.0);
        if (u_noise_method == NOISE_METHOD_PRNG) {
            xi.x = prng();
            xi.y = prng();
        }
        else if (u_noise_method == NOISE_METHOD_HEITZ_BLUENOISE) {
            xi.x = heitz_sample();
            xi.y = heitz_sample();
        }

        vec3 H = GGX_importance_sample(xi, state.roughness, N);
        L = 2.0 * dot(V, H) * H - V;

        return specular_brdf(V, N, L, H, state, pdf);
    }

    L = vec3(0.0);
    pdf = 0.0;
    return vec3(0.0);
}

/*
    Generate ray from camera position to screen position.
*/
Ray generate_ray(vec2 uv) {
    vec3 screen_world = u_camera.center +
                        u_camera.u * uv.x +
                        u_camera.v * uv.y;

    return Ray(
        u_camera.position,
        normalize(screen_world - u_camera.position)
    );
}

/*
    Reproject position in world space back to old camera and get UV.
*/
vec3 reproject(vec3 world_pos) {
    vec3 delta = world_pos - u_prev_camera.position;
    vec3 delta_n = normalize(delta);

    vec3 camera_dir = normalize(u_prev_camera.center - u_prev_camera.position);
	vec3 right = normalize(u_prev_camera.u);
    vec3 up = normalize(u_prev_camera.v);

    // Too close
    float d = dot(camera_dir, delta_n);
    if (d < EPSILON) {
        return vec3(0.0, 0.0, -1.0);
    }
    d = 1.0 / d;

    delta = delta_n * d - camera_dir;

    float x = dot(delta, right) / length(u_prev_camera.u);
    float y = dot(delta, up) / length(u_prev_camera.v);
    vec2 uv = vec2(x, y);

    // [-1, 1] -> [0, 1]
    uv = uv * 0.50 + 0.50;

    return vec3(uv, 1.0);
}


void sample_sun_cone(
    vec3 sun_dir,
    float sun_angular_radius,
    out vec3 world_dir,
    out float pdf
) {
    // There must be a better way...

    // We want to sample the sun disk over the hemisphere which creates a cone
    
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

    float cos_theta_max = cos(sun_angular_radius);
    float cos_theta = 1.0 - r0 * (1.0 - cos_theta_max);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    float phi = 2.0 * PI * r1;

    // Spherical
    vec3 local = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    // Should I use the same basis in GGX importance sampler?
    vec3 w = normalize(sun_dir);
    vec3 up = abs(w.z) > 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 tangent = normalize(cross(up, w));
    vec3 bitangent = cross(w, tangent);

    world_dir = normalize(
        tangent * local.x +
        bitangent * local.y +
        w * local.z
    );

    float solid_angle = 2.0 * PI * (1.0 - cos_theta);
    pdf = 1.0 / solid_angle;
}

bool is_ray_occluded(vec3 pos, vec3 dir) {
    HitInfo hitinfo = dda(Ray(pos, dir));
    return hitinfo.hit;
}

/*
    Path-trace a single ray and gather radiance information.
*/
vec3 pathtrace(Ray ray, out HitInfo primary_hit, out int total_bounces) {
    vec3 radiance = vec3(0.0); // Final ray color
    vec3 radiance_delta = vec3(1.0); // Accumulated multiplier

    for (total_bounces = 0; total_bounces < u_bounces; total_bounces++) {

        HitInfo hitinfo;

        /* This is only for material preview, will probably be removed. */
        if (u_exp_raymarch != 0) {
            hitinfo = raymarch(ray);
        }
        else {
            hitinfo = dda(ray);
        }

        /*
            In temporal reprojection use the first valid ray as the primary ray.
            TODO: Can this be improved? Maybe average all bounces together?
        */
        if (total_bounces == 0) {
            primary_hit = hitinfo;
        }

        // Ray did not hit anything, sample sky
        if (!hitinfo.hit) {
            float cos_angle = dot(ray.dir, u_sun_direction);
            float cos_theta_max = cos(u_sun_angular_radius);

            // We shouldn't show the sun in the sky to avoid double-counting lights
            // if NEE is enabled, BRDF shouldn't reach the sun.
            bool show_sun = (cos_angle >= cos_theta_max) &&
                            (!u_enable_nee || total_bounces == 0);

            if (show_sun) {
                radiance += u_sun_radiance * radiance_delta;
                break;
            }
            else {
                vec3 sky_color;
                if (u_enable_sky_texture) {
                    sky_color = texture(s_sky, uv_project_sphere(ray.dir)).rgb;

                    // Sky texture is already tonemapped
                    // sky_color = pow(sky_color, vec3(2.2));
                }
                else {
                    sky_color = u_sky_color;
                }

                radiance += sky_color * radiance_delta;
                break;
            }
        }

        vec3 N = normalize(hitinfo.normal);
        vec3 V = normalize(-ray.dir);

        /******************************

            Prepare surface material

         ******************************/

        /*
            0 -> Top
            1 -> Bottom
            2 -> Side
        */
        float surface_id = 0.0;
        if (hitinfo.normal.y > 0.0) surface_id = 0.0;
        else if (hitinfo.normal.y < 0.0) surface_id = 1.0;
        else surface_id = 2.0;

        float atlas_w = 1.0 / 3.0;
        float atlas_h = 1.0 / 8.0; // TODO: Pass block row count as uniform (or pass 1/h)

        vec2 atlas_uv = hitinfo.face_uv;
        atlas_uv.x = atlas_uv.x * atlas_w + float(surface_id) * atlas_w;
        atlas_uv.y = atlas_uv.y * atlas_h + float(hitinfo.block_id - 1) * atlas_h;

        vec3 albedo = texture(s_albedo_atlas, atlas_uv).rgb;
        vec3 emissive = texture(s_emissive_atlas, atlas_uv).rgb * EMISSIVE_MULT;
        float metallic = texture(s_metallic_atlas, atlas_uv).r;
        float roughness = texture(s_roughness_atlas, atlas_uv).r;

        Material material = Material(
            albedo,
            emissive,
            metallic,
            roughness,
            DIELECTRIC_BASE_REFLECTANCE
        );

        if (u_exp_raymarch != 0) {
            material = u_exp_material;
        }

        /******************************

           NEE (Next Event Estimation)

         ******************************/

        if (u_enable_nee) {
            // TODO: Is this the best bethod to sample sun?
            vec3 sun_world_dir;
            float sun_pdf;
            sample_sun_cone(
                u_sun_direction,
                u_sun_angular_radius,
                sun_world_dir,
                sun_pdf
            );

            if (!is_ray_occluded(hitinfo.point + N * EPSILON, sun_world_dir)) {
                float NoL = max(dot(N, sun_world_dir), 0.0);

                BRDFState state = prepare_brdf(material);
                vec3 H = normalize(V + sun_world_dir);

                vec3 nee_brdf = vec3(0.0);
                float nee_pdf = 0.0; // Not used in NEE, light's PDF is used instead 
                if (state.lobe < state.diffuse_weight) {
                    nee_brdf = diffuse_brdf(V, N, sun_world_dir, state, nee_pdf);
                }
                else {
                    nee_brdf = specular_brdf(V, N, sun_world_dir, H, state, nee_pdf);
                }

                radiance += radiance_delta * nee_brdf * u_sun_radiance * NoL / sun_pdf;
            }
        }

        /******************************

            Indirect lighting (BRDF)

         ******************************/

        vec3 L;
        float pdf;
        vec3 brdf = sample_brdf(N, V, material, L, pdf);

        // Current surface emission
        if (length(material.emissive) > 0.0) {
            radiance += material.emissive * radiance_delta;
            // TODO: to break; or not break; ?
        }

        // Absorption
        if (pdf > 0.0) {
            // BRDF is already multiplied by NoL
            radiance_delta *= brdf / pdf;
        }

        // Spawn new ray from the BRDF reflection
        ray = Ray(hitinfo.point + N * EPSILON, L);

        /*
            TODO: Adjust to new BRDF & NEE.
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

    int total_bounces = 0;

    float u_ray_countf = float(u_ray_count);
    for (int sample_i = 0; sample_i < u_ray_count; sample_i++) {

        // Initialize PRNGs
        int temporal_frame_i = accumulations[pixel.x + pixel.y * int(u_resolution.x)];
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
            vec2 jitter_amount = 1.0 / u_resolution;
            jitter_amount *= 2.0;
            ray_pos += vec2(prng() - 0.5, prng() - 0.5) * jitter_amount;
        }

        Ray ray = generate_ray(ray_pos);

        int sample_bounces = 0;
        vec3 radiance = pathtrace(ray, primary_hit, sample_bounces);

        final_radiance += radiance / u_ray_countf;
        total_bounces += sample_bounces;
    }

    float normalized_bounces = (float(total_bounces) / float(u_ray_count) / float(u_bounces));
    f_bounces = vec4(vec3(normalized_bounces), 1.0);

    vec3 final_color = final_radiance;

    f_normal = vec4(0.0);

    float curr_depth = length(primary_hit.point - u_camera.position);

    /*
        Temporal accumulation using reprojection:

        The idea is caching the last frame's render, gathering current frame's
        geometry and reprojecting back to previous frame. Then we can accumulate
        frames and do progressive rendering as usual. But now we have the ability
        to move the camera around while the pixels are still converging.

        However, we also need to limit the amount of accumulation so that very
        old frames do not have as much impact as the newer ones. This leads to
        some added noise and sharp reflections being delayed.

        I'm sure there are solutions for these problems but I'm fine with my
        current implementation.
    */
    float max_accumulation_frames = 16.0;
    if (u_enable_accumulation) {
        vec3 prev_uv = reproject(primary_hit.point);

        if (primary_hit.hit &&
            prev_uv.z > 0.0 &&
            (prev_uv.x > 0.0 && prev_uv.x < 1.0 && prev_uv.y > 0.0 && prev_uv.y < 1.0)
        ) {
            f_normal = vec4(primary_hit.normal, curr_depth);

            /*
                To match previous and current frame's pixels, I only use depth
                and normal information.

                TODO: Find better matching algorithms.
            */

            // rgb -> normal  alpha -> depth
            vec4 _previous_normal_sample = texture(s_previous_normal, prev_uv.xy);
            vec3 previous_normal = normalize(_previous_normal_sample.rgb);
            float previous_depth = _previous_normal_sample.a;

            // Depth is relative to each scene, it's not linear
            float normal_diff = dot(primary_hit.normal, previous_normal);
            float rel_depth_diff = abs(curr_depth - previous_depth) / max(curr_depth, previous_depth);

            // Very arbitrary values I found by playing around ...
            float normal_threshold = 0.97;
            float depth_threshold = 0.12;

            if (normal_diff > normal_threshold && rel_depth_diff < depth_threshold) {
                vec3 previous_color = texture(s_previous_frame, prev_uv.xy).rgb;

                // Temporal blending weight
                int acc = accumulations[pixel.x + pixel.y * int(u_resolution.x)];
                float capped_frame = min(float(acc), max_accumulation_frames);
                float weight = 1.0 / (capped_frame + 1.0);

                // Blend current and reprojected colors
                final_color = mix(previous_color, final_color, weight);

                accumulations[pixel.x + pixel.y * int(u_resolution.x)] += 1;
            }
            else {
                final_color = final_color;
                accumulations[pixel.x + pixel.y * int(u_resolution.x)] = 0;
            }
        }
        else {
            // No valid history reset accumulation
            final_color = final_color;
            accumulations[pixel.x + pixel.y * int(u_resolution.x)] = 0;
        }
    }

    f_color = vec4(final_color, luminance(final_color));

    if (issue) {
        f_color = vec4(1000.0, 0.0, 0.0, 1000.0);
    }
}
