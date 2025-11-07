/*

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

*/

/* OMIT START */
precision highp float;
/* OMIT END */

#ifndef TYPES_H
#define TYPES_H


/*
    Basic pin-hole camera model.

    position
        Position of camera in world space
    center
        Front + position
    u
        Horizontal image plane vector
    v
        Vertical image plane vector
*/
struct Camera {
    vec3 position;
    vec3 center;
    vec3 u;
    vec3 v;
};

/*
    Ray definition.

    origin
        Origin position of ray in world space
    dir
        Normalized direction of ray
*/
struct Ray {
    vec3 origin;
    vec3 dir;
};

/*
    Ray vs voxel collision information.

    hit
        Collision with ray happened?
    point
        Collision point on the surface in world space
    normal
        Normal of the collision surface
    inside
        Started inside the medium?
    block_id
        ID of the block type
    face_uv
        Local UV coordinate on the hit face
*/
struct HitInfo {
    bool hit;
    vec3 point;
    vec3 normal;
    bool inside;
    int block_id;
    vec2 face_uv;
};

/*
    Physically-based surface material definition.

    albedo
        Base color. If it's a texture map, it should have minimal or preferably
        no shadows because that should be added with an AO map later.
        In range [0, 1].
    emissive
        Emission color.
        In range [0, infinity].
    metallic
        Metallness. 0 = Dielectric, 1 = Metallic (conductor).
        Values in between are interpolated, so it doesn't really make
        sense to not have binary values.
        In range [0, 1].
    roughness
        Perceptual microfacet roughness for specular reflection and transmission.
        In range [0, 1].
    reflectance
        Base specular reflectance (f0).
        In range [0, 1].
    glass
        Transmission weight. 0 = Opaque, 1 = Fully transmissive.
        In range [0, 1].
    ior
        Index of Refraction. Air has an IOR of 1.0.
        In range [1, infinity].
*/
struct Material {
    vec3 albedo;
    vec3 emissive;
    float metallic;
    float roughness;
    float reflectance;
    float glass;
    float ior;
};


#endif // TYPES_H