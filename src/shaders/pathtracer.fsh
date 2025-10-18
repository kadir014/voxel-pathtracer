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


in vec2 v_uv;
out vec4 f_color;


uniform int u_ray_count;
uniform int u_bounces;
uniform int u_noise_method;
uniform vec2 u_resolution;

uniform sampler2D s_bluenoise;


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
    bool hit;
    vec3 point;
    vec3 normal;
    Material material;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Triangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    vec3 normal;
    Material material;
};

struct Quad {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    vec3 v3;
    Material material;
};


/*
    Sphere x Ray intersection function by Inigo Quilez
    https://iquilezles.org/articles/intersectors/
*/
HitInfo sphere_x_ray(Sphere sphere, Ray ray) {
    HitInfo empty_hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
    );

    vec3 oc = ray.origin - sphere.center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float h = b * b - c;

    // No intersection
    if (h < 0.0) return empty_hitinfo;
    h = sqrt(h);

    vec2 t = vec2(-b-h, -b+h);

    // Ray does NOT intersect the sphere
    if (t.y < 0.0 ) return empty_hitinfo;

    float d = 0.0;
    if (t.x < 0.0) d = t.y; // Ray origin inside the sphere, t.y is intersection distance
    else d = t.x; // Ray origin outside the sphere, t.x is intersection distance

    vec3 intersection = ray.origin + ray.dir * d;
    vec3 normal = normalize(intersection - sphere.center);

    return HitInfo(true, intersection, normal, sphere.material);
}

/*
    Triangle x Ray intersection function by Inigo Quilez
    https://iquilezles.org/articles/intersectors/
*/
HitInfo triangle_x_ray(Triangle triangle, Ray ray) {
    HitInfo empty_hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
    );

    vec3 v1v0 = triangle.v1 - triangle.v0;
    vec3 v2v0 = triangle.v2 - triangle.v0;
    vec3 rov0 = ray.origin - triangle.v0;
    vec3 n = cross(v1v0, v2v0);
    vec3 q = cross(rov0, ray.dir);
    float d = 1.0 / dot(ray.dir, n);
    float u = d * dot(-q, v2v0);
    float v = d * dot(q, v1v0);
    float t = d * dot(-n, rov0);

    if (u < 0.0 || v < 0.0 || (u + v) > 1.0 ) return empty_hitinfo;
    if (t <= 0.0) return empty_hitinfo;

    vec3 intersection = ray.origin + t * ray.dir;
    vec3 normal = triangle.normal;
    if (dot(normal, ray.dir) > 0.0) normal = -normal;

    return HitInfo(true, intersection, normal, triangle.material);
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

int bluenoise_counter;
float bluenoise2() {
    int x = int(bluenoise_seed.x * u_resolution.x);
    int y = int(bluenoise_seed.y * u_resolution.y);
    int xi = (x + (bluenoise_counter * 37)) & (BLUENOISE_SIZE - 1);
    int yi = (y + (bluenoise_counter * 57)) & (BLUENOISE_SIZE - 1);
    bluenoise_counter++;
    vec4 bluenoise_sample = texture(s_bluenoise, vec2(xi, yi) / vec2(BLUENOISE_SIZE, BLUENOISE_SIZE));
    return fract(bluenoise_sample).r;
}

int current_sample_idx;
float bluenoise3(int dim) {
    // int x = int(bluenoise_seed.x * u_resolution.x);
    // int y = int(bluenoise_seed.y * u_resolution.y);
    // int xi = (x + current_sample_idx * 73 + dim * 17) & (BLUENOISE_SIZE - 1);
    // int yi = (y + current_sample_idx * 91 + dim * 23) & (BLUENOISE_SIZE - 1);
    
    // vec4 bluenoise_sample = texture(s_bluenoise, vec2(xi, yi) / vec2(BLUENOISE_SIZE, BLUENOISE_SIZE));
    // return fract(bluenoise_sample).r;

    ivec2 pixel = ivec2(bluenoise_seed * u_resolution);
    float seed = texelFetch(s_bluenoise, ivec2(pixel) & (BLUENOISE_SIZE-1), 0).r;
    float r = fract(seed + float(current_sample_idx) * 0.618034 + float(dim) * 0.0426727);
    return r;
}


vec3 random_in_unit_sphere() {
    float r0 = 0.0;
    float r1 = 0.0;
    if (u_noise_method == 1) {
        r0 = prng();
        r1 = prng();
    }
    else if (u_noise_method == 2) {
        r0 = bluenoise3(0);
        r1 = bluenoise3(1);
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

    specular = (specular_chance < hitinfo.material.specular_percentage) ? 1.0 : 0.0;

    vec3 diffuse_ray_dir = normalize(hitinfo.normal + random_in_unit_sphere());
    vec3 specular_ray_dir = reflect(ray.dir, hitinfo.normal);
    specular_ray_dir = normalize(mix(specular_ray_dir, diffuse_ray_dir, hitinfo.material.roughness * hitinfo.material.roughness));

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
    Cast the ray into the scene and gather collided objects.
*/
HitInfo cast_ray(Ray ray, Sphere[6] spheres, Triangle[22] tris, int skip_i) {
    float min_depth = BIG_VALUE;
    HitInfo min_hitinfo = HitInfo(false, vec3(0.0), vec3(0.0), Material(vec3(0.0), vec3(0.0), 0, vec3(0.0), 0.0));
    
    for (int i = 0; i < 6; i++) {
        if (i == skip_i) {continue;}

        HitInfo hitinfo = sphere_x_ray(spheres[i], ray);

        if (hitinfo.hit) {
            float dist = distance(hitinfo.point, ray.origin);

            if (dist < min_depth) {
                min_depth = dist;
                min_hitinfo = hitinfo;
            }
        }
    }

    for (int i = 0; i < 22; i++) {
        if (i == skip_i) {continue;}

        HitInfo hitinfo = triangle_x_ray(tris[i], ray);

        if (hitinfo.hit) {
            float dist = distance(hitinfo.point, ray.origin);

            if (dist < min_depth) {
                min_depth = dist;
                min_hitinfo = hitinfo;
            }
        }
    }

    return min_hitinfo;
}

/*
    Path-trace a single ray and gather radiance information.
*/
vec3 pathtrace(Ray ray, Sphere[6] spheres, Triangle[22] tris) {
    vec3 radiance = vec3(0.0); // Final ray color
    vec3 radiance_delta = vec3(1.0); // Accumulated multiplier

    for (int bounce = 0; bounce < u_bounces; bounce++) {
        
        HitInfo hitinfo = cast_ray(ray, spheres, tris, -1);

        // Ray did not hit anything, sample sky
        if (!hitinfo.hit) {
            //vec3 sky_color = srgb_to_rgb_approx(texture(s_sky, uv_project_sphere(nray.dir)).rgb);
            vec3 sky_color = vec3(0.0);
            radiance += sky_color * radiance_delta;
            break;
        }

        float specular = 0.0;
        ray = scatter(ray, hitinfo, specular);

        radiance += hitinfo.material.emissive * radiance_delta;
        radiance_delta *= mix(hitinfo.material.color, hitinfo.material.specular_color, specular);

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

    // Generate scene
    Sphere[6] spheres = Sphere[](
        Sphere(
            vec3(-10.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.0)
        ),
        Sphere(
            vec3(-3.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.333)
        ),
        Sphere(
            vec3(3.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.667)
        ),
        Sphere(
            vec3(10.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 1.0)
        ),

        Sphere(
            vec3(0.0, 13.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        Sphere(
            vec3(0.0, 14.0, 25.0),
            1.0,
            Material(vec3(0.0), vec3(1.0) * 2.0, 0.0, vec3(0.0), 0.0)
        )
    );

    Quad[11] quads = Quad[](
        // Floor
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 15.0),
            vec3(-15.0, -15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Ceiling
        Quad(
            vec3(-15.0, 15.0, 45.0),
            vec3( 15.0, 15.0, 45.0),
            vec3( 15.0, 15.0, 15.0),
            vec3(-15.0, 15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.5), 0.1)
        ),

        // Back wall
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 45.0),
            vec3( 15.0,  15.0, 45.0),
            vec3(-15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Front wall
        Quad(
            vec3(-15.0, -15.0, 15.0),
            vec3( 15.0, -15.0, 15.0),
            vec3( 15.0,  -15.0, 15.0),
            vec3(-15.0,  -15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Left wall
        Quad(
            vec3(15.0, -15.0, 45.0),
            vec3(15.0, -15.0, 15.0),
            vec3(15.0,  15.0, 15.0),
            vec3(15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Right wall
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3(-15.0, -15.0, 15.0),
            vec3(-15.0,  15.0, 15.0),
            vec3(-15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Light
        Quad(
            vec3(-13.0, -15.0, 44.9),
            vec3( -11.0, -15.0, 44.9),
            vec3( -11.0,  15.0, 44.9),
            vec3(-13.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(1.0, 0.0, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(-7.0, -15.0,  44.9),
            vec3( -5.0, -15.0, 44.9),
            vec3( -5.0,  15.0, 44.9),
            vec3(-7.0,  15.0,  44.9),
            Material(vec3(0.0), vec3(1.0, 0.5, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(-1.0, -15.0, 44.9),
            vec3( 1.0, -15.0, 44.9),
            vec3( 1.0,  15.0, 44.9),
            vec3(-1.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(0.0, 1.0, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(5.0, -15.0, 44.9),
            vec3(7.0, -15.0, 44.9),
            vec3(7.0,  15.0, 44.9),
            vec3(5.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(0.0, 0.5, 1.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(13.0, -15.0, 44.9),
            vec3(11.0, -15.0, 44.9),
            vec3(11.0,  15.0, 44.9),
            vec3(13.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(0.5, 0.0, 1.0) * 4.0, 0.0, vec3(0.0), 0.0)
        )
    );

    // Calculate triangles from quads
    Triangle[22] tris;
    int j = 0;
    for (int i = 0; i < 22; i += 2) {
        Triangle tri0 = Triangle(quads[j].v0, quads[j].v1, quads[j].v2, vec3(0.0), quads[j].material);
        Triangle tri1 = Triangle(quads[j].v2, quads[j].v3, quads[j].v0, vec3(0.0), quads[j].material);
        tri0.normal = normalize(cross(tri0.v1 - tri0.v0, tri0.v2 - tri0.v0));
        tri1.normal = normalize(cross(tri1.v1 - tri1.v0, tri1.v2 - tri1.v0));
        tris[i] = tri0;
        tris[i + 1] = tri1;
        j++;
    }

    // prng_state = wang_hash(
    //     uint(v_uv.x * u_resolution.x) * 73856093u ^
    //     uint(v_uv.y * u_resolution.y) * 19349663u
    // );

    vec2 pixel = v_uv * u_resolution;

    bluenoise_seed = v_uv;
    bluenoise_counter = 0;

    float u_ray_countf = float(u_ray_count);
    for (int i = 0; i < u_ray_count; i++) {
        current_sample_idx = i;

        uint s = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + uint(i) * 26699u;
        prng_state = wang_hash(s);

        // Anti-aliasing
        // float rx = (prng(prng_state) * 2.0 - 1.0) / 790.0;
        // float ry = (prng(prng_state) * 2.0 - 1.0) / 790.0;
        // vec2 pos = v_uv * 2.0 - 1.0 + vec2(rx, ry);

        vec2 pos = v_uv * 2.0 - 1.0;

        Ray ray = generate_ray(pos);

        vec3 radiance = pathtrace(ray, spheres, tris);
        
        final_radiance += radiance / u_ray_countf;
    }

    f_color = vec4(final_radiance, 1.0);
}