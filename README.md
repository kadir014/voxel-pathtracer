# voxel-pathtracer
A work-in-progress voxel pathtracer prototype in a sandbox environment.

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/cornell.jpg" width=400>



# Features
- Real-time
- Global illumination via Monte Carlo Path Tracing
- Voxel world traversal accelerated with DDA
- Low-Discrepancy Bluenoise sampler
- Temporal frame accumulation with reprojecting
- Post-processing
  - Filmic tonemapping
  - Color grading
  - Upscaling
- Physically-based material pipeline with texture maps
  - Albedo, emission, metallic, roughness, reflectance.
- Custome skydome
- Russian Roulette path termination
- Anti-aliasing
  - Jitter sampling
  - FXAA

**Roadmap:**
- Disney BRDF
- Procuderal sky / atmosphere model
- Octree acceleration
- Transparent objects & transmission
- Volumetrics
- Volumetric clouds (realistic & stylistic)
- Denoising
- Next event estimation (NEE)
- Multiple importance sampling (MIS)



# Running
You need Python 3.11+. After cloning the repo, install required packages:
```shell
$ python -m pip install -r requirements.txt
```
And then just run `main.py`. You can edit `src/common.py` to adjust common settings.
```shell
$ python main.py
```


# Resources & References
- E. Heitz et al, [A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space](https://eheitzresearch.wordpress.com/762-2/)
- M. Pharr, W. Jakob, and G. Humphreys, ["Physically Based Rendering" book](https://www.pbr-book.org/4ed/contents)
- P. Shirley, T. Black, S. Hollasch, ["Ray Tracing in One Weekend" book series](https://raytracing.github.io/)
- Brian Karis, [Real Shading in Unreal Engine 4](https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf)
- Brent Burley, [Physically Based Shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)
- R. Guy, M. Agopian, [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.md.html)
- Marco Alamia, [Physically Based Rendering - Cook-Torrance](http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx)
- Jacco Bikker, [Reprojection in a Ray Tracer](https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/)
- Alan Wolfe, ["Casual Shadertoy Path Tracing" blog series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/)
- Scratchapixel, [Ray tracing articles](https://www.scratchapixel.com/)
- Academy Software, [OpenPBR specification](https://academysoftwarefoundation.github.io/OpenPBR/)



# License
[MIT](LICENSE) © Kadir Aksoy