# voxel-pathtracer
A work-in-progress real-time voxel path tracer implementing physically-based rendering.

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/gallery/small_thumbnail.jpg" width=400>



# Features
- **Real-Time Voxel Path Tracing**
  - Monte Carlo global illumination with multiple bounces
  - Accelerated voxel traversal via DDA algorithm
  - Russian Roulette termination for unbiased energy conservation

- **Sampling & Image Stability**
  - Low-discrepancy blue noise sampling
  - Temporal accumulation with reprojection
  - Next Event Estimation (NEE) for explicit sun sampling
  - Multiple anti-aliasing methods like FXAA and subpixel jitter

- **Physically-Based Materials**
  - Unified PBR pipeline inspired by UE4's model
  - Real-time & artistic efficiency
  - Texture maps for each property such as albedo, metallic, roughness, ...

- **Post-Processing Pipeline**
  - Filmic tonemapping
  - Color grading
  - Chromatic aberration
  - Upsampling

- **Environment**
  - Analytical Preetham sky model with sun disk
  - Custom skydome texture
  - Emissive materials as light sources

### Roadmap
- Volumetric clouds (realistic & stylistic)
- Octree acceleration for world traversal
- Transparent objects & transmission (Disney BSDF?)
- Volumetrics
- Denoising
- Multiple importance sampling (MIS) with NEE



# Running
You need Python 3.11+. After cloning the repo, install required packages:
```shell
$ python -m pip install -r requirements.txt
```
And then just run `main.py`. You can edit `src/common.py` to adjust common settings.
```shell
$ python main.py
```


# Gallery
<details>
<summary>Click here to see images (loading may take a second)</summary>

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/gallery/cornell_box.png">

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/gallery/couch.png">

</details>
<br>


# Resources & References
- E. Heitz et al, [A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space](https://eheitzresearch.wordpress.com/762-2/)
- M. Pharr, W. Jakob, and G. Humphreys, ["Physically Based Rendering" book](https://www.pbr-book.org/4ed/contents)
- P. Shirley, T. Black, S. Hollasch, ["Ray Tracing in One Weekend" book series](https://raytracing.github.io/)
- TU Wien, [2021 Rendering Lectures](https://www.youtube.com/watch?v=FU1dbi827LY&list=PLmIqTlJ6KsE2yXzeq02hqCDpOdtj6n6A9&index=10)
- Brian Karis, [Real Shading in Unreal Engine 4](https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf)
- Brent Burley, [Physically Based Shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)
- A. J. Preetham et al, [A Practical Analytic Model for Daylight](https://courses.cs.duke.edu/cps124/spring08/assign/07_papers/p91-preetham.pdf)
- R. Guy, M. Agopian, [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.md.html)
- Marco Alamia, [Physically Based Rendering - Cook-Torrance](http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx)
- Jacco Bikker, [Reprojection in a Ray Tracer](https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/)
- Alan Wolfe, ["Casual Shadertoy Path Tracing" blog series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/)
- Scratchapixel, [Ray tracing articles](https://www.scratchapixel.com/)
- Academy Software, [OpenPBR specification](https://academysoftwarefoundation.github.io/OpenPBR/)



# License
[MIT](LICENSE) © Kadir Aksoy