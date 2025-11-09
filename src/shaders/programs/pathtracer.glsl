/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/*
    pathtracer.fsh
    --------------
    Path-traced global illumination shader.
*/

#version 460
#extension GL_ARB_shading_language_include: enable


in vec2 v_uv;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_normal;
layout(location = 2) out vec4 f_bounces;
layout(location = 3) out vec4 f_position;
layout(location = 4) out vec4 f_albedo;


uniform int u_ray_count;
uniform int u_bounces;
uniform int u_noise_method;
uniform vec2 u_resolution;
uniform float u_voxel_size;
uniform bool u_enable_roulette;
uniform bool u_enable_sky_texture;
uniform bool u_enable_nee;
uniform bool u_enable_accumulation;
uniform int u_antialiasing;
uniform int u_exp_raymarch;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_radiance;
uniform float u_sun_angular_radius;
uniform float u_turbidity;

uniform sampler3D s_grid;
uniform sampler2D s_sky;
uniform sampler2D s_albedo_atlas;
uniform sampler2D s_emissive_atlas;
uniform sampler2D s_roughness_atlas;
uniform sampler2D s_metallic_atlas;
uniform sampler2D s_glass_atlas;
uniform sampler2D s_previous_frame;
uniform sampler2D s_previous_normal;

layout(std430, binding = 2) buffer AccLayout {
    int accumulations[]; // Logical width x Logical height
};


vec3 issue_color;
bool issue;


#include "../libs/common.glsl"
#include "../libs/types.glsl"
#include "../libs/prng.glsl"
#include "../libs/heitz.glsl"
#include "../libs/preetham.glsl"
#include "../libs/microfacet.glsl"
#include "../libs/bsdf.glsl"
#include "../libs/bicubic.glsl"


uniform Camera u_camera;
uniform Camera u_prev_camera;

// Material preview feature only, will be removed
uniform Material u_exp_material;


HitInfo dda(Ray ray) {
    HitInfo hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        false,
        0,
        vec2(0.0)
    );

    vec3 voxel = ray.origin / u_voxel_size;
    voxel.x = floor(voxel.x);
    voxel.y = floor(voxel.y);
    voxel.z = floor(voxel.z);

    vec4 first_sample = texelFetch(s_grid, ivec3(voxel), 0);
    hitinfo.inside = first_sample.r > 0.0;

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

        // Connected voxels
        // Useful for glass
        if (voxel_sample.r == first_sample.r) continue;

        if (hitinfo.inside || voxel_sample.r > 0.0) {
            hitinfo.hit = true;

            hitinfo.point = ray.origin + ray.dir * hit_t;

            //if (hitinfo.inside) hitinfo.normal = -hitinfo.normal;

            if (hitinfo.inside) {
                hitinfo.block_id = int(first_sample.r * 255.0);
            }
            else {
                hitinfo.block_id = int(voxel_sample.r * 255.0);
            }

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

   //return sdf_sphere(pos, vec3(0.0), 1.0);
   //return sdf_round_box(pos, vec3(0.0), vec3(1.0), 0.5);
   return sdf_box(pos, vec3(0.0), vec3(1.0));
}

vec3 estimate_normal(vec3 p) {
    return normalize(vec3(
        sdf(vec3(p.x + EPSILON, p.y, p.z)) - sdf(vec3(p.x - EPSILON, p.y, p.z)),
        sdf(vec3(p.x, p.y + EPSILON, p.z)) - sdf(vec3(p.x, p.y - EPSILON, p.z)),
        sdf(vec3(p.x, p.y, p.z  + EPSILON)) - sdf(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

HitInfo raymarch(Ray ray) {
    vec3 curr_pos = ray.origin;
    bool inside = sdf(curr_pos) < 0.0;

    for (int i = 0; i < 500; i++) {
    	float dist = sdf(curr_pos);

        if (abs(dist) < EPSILON) {
            vec3 normal = estimate_normal(curr_pos);

            if (inside) normal = -normal;

            return HitInfo(
                true,
                curr_pos,
                normal,
                inside,
                0,
                vec2(0.0, 0.0)
            );
        }

        curr_pos += ray.dir * abs(dist);
    }

    return HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        false,
        0,
        vec2(0.0)
    );
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


Material material_from_hitinfo(HitInfo hitinfo) {
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
    float atlas_h = 1.0 / 9.0; // TODO: Pass block row count as uniform (or pass 1/h)

    vec2 atlas_uv = hitinfo.face_uv;
    atlas_uv.x = atlas_uv.x * atlas_w + float(surface_id) * atlas_w;
    atlas_uv.y = atlas_uv.y * atlas_h + float(hitinfo.block_id - 1) * atlas_h;

    vec3 albedo = texture(s_albedo_atlas, atlas_uv).rgb;
    vec3 emissive = texture(s_emissive_atlas, atlas_uv).rgb * EMISSIVE_MULT;
    float roughness = texture(s_roughness_atlas, atlas_uv).r;
    float metallic = texture(s_metallic_atlas, atlas_uv).r;
    float glass = texture(s_glass_atlas, atlas_uv).r;

    return Material(
        albedo,
        emissive,
        metallic,
        roughness,
        DIELECTRIC_BASE_REFLECTANCE,
        glass,
        1.5
    );
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

    // If we encounter a glass, disable this
    bool allow_nee = u_enable_nee;

    for (total_bounces = 0; total_bounces < u_bounces; total_bounces++) {

        /******************************

                 Trace the ray

         ******************************/

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

        /******************************

              Environment sampling

         ******************************/

        if (!hitinfo.hit) {
            float cos_angle = dot(ray.dir, u_sun_direction);
            float cos_theta_max = cos(u_sun_angular_radius);

            // We shouldn't show the sun in the sky to avoid double-counting lights
            // if NEE is enabled, BSDF shouldn't reach the sun.
            bool show_sun = (cos_angle >= cos_theta_max) &&
                            (!allow_nee || total_bounces == 0);

            if (show_sun) {
                //float solid_angle = TAU * (1.0 - cos_angle) * 1.0;
                //radiance += u_sun_radiance / solid_angle * radiance_delta;
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
                    sky_color = preetham_sky(u_sun_direction, ray.dir, u_turbidity) * 0.052;
                }

                radiance += sky_color * radiance_delta;
                break;
            }

        //     float cos_angle = dot(ray.dir, u_sun_direction);
        //     float cos_theta_max = cos(u_sun_angular_radius);
        //     float epsilon = 0.003; // smooth edge

        //     // Smooth blend between sun and sky
        //     vec3 sun_world;
        //     float sun_pdf;
        //     sample_sun_cone(ray.dir, u_sun_angular_radius, sun_world, sun_pdf);
        //     float sun_blend = dot(sun_world, u_sun_direction);

        //     // Match NEE cone radiance
        //     float solid_angle = TAU * (1.0 - cos_angle);
        //     vec3 sun_radiance_per_sa = u_sun_radiance / solid_angle;

        //     // Combine sun and sky
        //    vec3 sky_color;
        //     if (u_enable_sky_texture) {
        //         sky_color = texture(s_sky, uv_project_sphere(ray.dir)).rgb;
        //         // Sky texture is already tonemapped
        //         // sky_color = pow(sky_color, vec3(2.2));
        //     }
        //     else {
        //         sky_color = preetham_sky(u_sun_direction, ray.dir, u_turbidity) * 0.052;
        //         sky_color *= vec3(0.00001);
        //     }

        //     //radiance += vec3(sun_blend) + sky_color;
        //     //break;

        //     radiance += sun_radiance_per_sa * sun_blend * radiance_delta;
        //     radiance += sky_color * (1.0 - sun_blend) * radiance_delta;

            break;
        }

        /******************************

            Prepare surface material

         ******************************/

        vec3 N = normalize(hitinfo.normal);
        vec3 V = normalize(-ray.dir);

        Material material = material_from_hitinfo(hitinfo);

        if (u_exp_raymarch != 0) {
            material = u_exp_material;
        }

        /******************************

           NEE (Next Event Estimation)

         ******************************/

        if (allow_nee) {
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

                BSDFState state = prepare_bsdf(material);
                vec3 H = normalize(V + sun_world_dir);

                vec3 nee_bsdf = vec3(0.0);
                float nee_pdf = 0.0;
                if (state.lobe < state.diffuse_weight) {
                    nee_bsdf = diffuse_brdf(V, N, sun_world_dir, state, nee_pdf);
                }
                else {
                    if (state.lobe < state.diffuse_weight + state.specular_weight) {
                        nee_bsdf = specular_brdf(V, N, sun_world_dir, H, state, nee_pdf);
                    }
                    else if (state.transmit_weight > 0.0) {
                        nee_bsdf = specular_btdf(V, N, sun_world_dir, H, hitinfo.inside, state, nee_pdf);
                    }
                }

                // Brings NaNs
                if (nee_pdf > 0.0) {
                    radiance += radiance_delta * nee_bsdf * u_sun_radiance / sun_pdf;
                }
                // else {
                //     issue = true;
                // }
            }
        }

        /******************************

            Indirect lighting (BSDF)

         ******************************/

        vec3 L;
        float pdf;
        vec3 bsdf = sample_bsdf(V, N, hitinfo.inside, material, L, pdf);

        // Current surface emission
        if (length(material.emissive) > 0.0) {
            radiance += material.emissive * radiance_delta;
            // TODO: to break; or not break; ?
        }

        // Absorption
        if (pdf > 0.0) {
            // BSDF radiance is already multiplied by NoL
            radiance_delta *= bsdf / pdf;
        }

        // Spawn new ray from the BSDF reflection
        // TODO: Find a better solution than multipling epsilon, otherwise glass doesn't work
        ray = Ray(hitinfo.point + (N * (EPSILON * 2.1)), L);

        // If BSDF sampling reached a transmissive surface, disable NEE so
        // BSDF can reach sky
        if (material.glass > 0.0) {
            allow_nee = false;
        }

        /*
            TODO: Adjust to new BSDF & NEE.
            Russian Roulette:
            As the throughput gets smaller, the ray is more likely to get terminated early.
            Survivors have their value boosted to make up for fewer samples being in the average.
        */
        if (u_enable_roulette && total_bounces > 3) {
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
    issue_color = vec3(1000.0, 0.0, 1000.0);

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

                HOWEVER! This does not work well with temporal accumulation and
                reprojection. It makes the edges noisier and everything blurry.
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

    Material mat = material_from_hitinfo(primary_hit);
    vec3 albedo = mat.albedo;
    // Demodulate albedo
    if (primary_hit.hit) {
        final_color = mix(final_color / albedo, vec3(0.0), equal(albedo, vec3(0.0)));
        f_albedo = vec4(albedo, 1.0);
    }
    else {
        f_albedo = vec4(1.0);
    }

    float curr_depth = length(primary_hit.point - u_camera.position);

    f_normal = vec4(0.0);
    if (primary_hit.hit) {
        f_normal = vec4(primary_hit.normal, curr_depth);
    }

    f_position = vec4(primary_hit.point, 1.0);

    /*
        Temporal accumulation using reprojection:

        The idea is caching the last frame's render, gathering current frame's
        geometry and reprojecting back to previous frame. Then we can accumulate
        frames and do progressive rendering as usual. But now we have the ability
        to move the camera around while the pixels are still converging.

        However, we also need to limit the amount of accumulation so that very
        old frames do not have as much impact as the newer ones. This leads to
        some added noise and sharp reflections being delayed.

        I'm sure there are smarter solutions for these problems but I'm fine
        with my current implementation.
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

            //issue = true;
            //issue_color = vec3(normal_diff);

            if (
                normal_diff > normal_threshold &&
                rel_depth_diff < depth_threshold
            ) {
                //vec3 previous_color = texture(s_previous_frame, prev_uv.xy).rgb;
                vec3 previous_color = texture_Bicubic(s_previous_frame, prev_uv.xy).rgb;

                if (!any(isnan(previous_color)) && !any(isinf(previous_color))) {
                    // Temporal blending weight
                    int acc = accumulations[pixel.x + pixel.y * int(u_resolution.x)];
                    float capped_frame = min(float(acc), max_accumulation_frames);
                    float weight = 1.0 / (capped_frame + 1.0);

                    // Blend current and reprojected colors
                    final_color = mix(previous_color, final_color, weight);

                    accumulations[pixel.x + pixel.y * int(u_resolution.x)] += 1;
                }
                else {
                    // Don't blend NaN history
                    final_color = final_color;
                    accumulations[pixel.x + pixel.y * int(u_resolution.x)] = 0;
                }
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
        f_color = vec4(issue_color, 1000.0);
    }
}
