<br>
<h1 align="center">Project Lyrae</h1>  
<p align="center">
  <a href="https://github.com/kadir014/project-lyrae/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img src="https://img.shields.io/badge/version-0.0.4-yellow">
</p>

<p align="center">
A physically-based real-time voxel path traced renderer.
<br><br>
<img src="https://raw.githubusercontent.com/kadir014/project-lyrae/refs/heads/main/data/gallery/thumb.png" width=550>
</p>



# Features
- **Real-Time Voxel Path Tracing**
  - Monte Carlo global illumination with multiple light bounces
  - Fast voxel traversal via DDA algorithm
  - Russian Roulette termination for unbiased energy conservation

- **Sampling & Image Stability**
  - Low-discrepancy blue noise sampling
  - Temporal accumulation with reprojection for noise reduction
  - Next Event Estimation (NEE) for explicit sun sampling
  - Edge avoiding À-Trous denoising
  - Multiple anti-aliasing methods like FXAA and subpixel jitter

- **Physically-Based Materials**
  - Unified physically-based shading pipeline
    - Diffuse and specular BRDF based on UE4's model
    - Specular BTDF based on B. Walter's paper
  - Designed for real-time & artistic efficiency
  - Texture maps for each material property

- **Post-Processing**
  - Filmic tonemapping
  - Color grading
  - Chromatic aberration
  - Upsampling

- **Environment**
  - Analytical Preetham sky model with sun disk
  - Custom skydome texture
  - Emissive materials as light sources

### Roadmap
- Normal, displacement, parallax and AO mapping
- Volumetric clouds (realistic & stylistic)
- Sparse octree acceleration for world traversal
- Volumetric mediums
  - Better atmospheric fog & implicit godrays
- Multiple importance sampling (MIS) with NEE



# Gallery
<details>
<summary>Click here to see images</summary>

<img src="https://raw.githubusercontent.com/kadir014/project-lyrae/refs/heads/main/data/gallery/cornell_box.png">

<img src="https://raw.githubusercontent.com/kadir014/project-lyrae/refs/heads/main/data/gallery/couch.png">

</details>
<br>


# Running
You need Python 3.11+. After cloning the repo, install required packages:
```shell
$ python -m pip install -r requirements.txt
```
And then just run `main.py`.
```shell
$ python main.py
```
You can edit `src/common.py` to adjust common settings such as window resolution.



# Resources & References
- E. Heitz et al — [A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space](https://eheitzresearch.wordpress.com/762-2/)
- H. Dammertz et al — [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)
- M. Pharr, W. Jakob, and G. Humphreys — ["Physically Based Rendering" book](https://www.pbr-book.org/4ed/contents)
- P. Shirley, T. Black, S. Hollasch — ["Ray Tracing in One Weekend" book series](https://raytracing.github.io/)
- TU Wien — [2021 Rendering Lectures](https://www.youtube.com/watch?v=FU1dbi827LY&list=PLmIqTlJ6KsE2yXzeq02hqCDpOdtj6n6A9&index=10)
- Brian Karis — [Real Shading in Unreal Engine 4](https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf)
- Brent Burley — [Physically Based Shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)
- B. Walter et al — [Microfacet Models for Refraction through Rough Surfaces](https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf)
- A. J. Preetham et al — [A Practical Analytic Model for Daylight](https://courses.cs.duke.edu/cps124/spring08/assign/07_papers/p91-preetham.pdf)
- R. Guy, M. Agopian — [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.md.html)
- Marco Alamia — [Physically Based Rendering - Cook-Torrance](http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx)
- Jacco Bikker — [Reprojection in a Ray Tracer](https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/)
- Alan Wolfe — ["Casual Shadertoy Path Tracing" blog series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/)
- Scratchapixel — [Ray tracing articles](https://www.scratchapixel.com/)
- Academy Software — [OpenPBR specification](https://academysoftwarefoundation.github.io/OpenPBR/)



# License
[MIT](LICENSE) © Kadir Aksoy

If you enjoy my projects, I'd greatly appreciate if you wanted to support me & my studies! ❤️

<a href="https://github.com/sponsors/kadir014"><img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#white"></a>
<a href="https://www.buymeacoffee.com/kadir014"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black"></a>