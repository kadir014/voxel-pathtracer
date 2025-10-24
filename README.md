# voxel-pathtracer
A work-in-progress voxel pathtracer prototype in a sandbox environment.

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/cornell.jpg" width=400>



# Features
- Real-time
- Global illumination via Monte Carlo Path Tracing
- Voxel world traversal accelerated with DDA
- Low-Discrepancy Bluenoise sampler
- Temporal frame accumulation with reprojecting
- Filmic tonemapping
- Color grading postprocessing
- Simple PBR-ish material pipeline with texture maps
  - Albedo, diffuse, emission, specular and roughness.
- Custome skydome
- Russian Roulette path termination
- Anti-aliasing with jitter sampling or FXAA

**Roadmap:**
- Disney BRDF
- Procuderal sky / atmosphere model
- Octree acceleration
- Transparent objects & transmission
- Volumetrics
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
- Jacco Bikker, [Reprojection in a Ray Tracer](https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/)
- Alan Wolfe, ["Casual Shadertoy Path Tracing" blog series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/)
- Scratchapixel, [Ray tracing articles](https://www.scratchapixel.com/)



# License
[MIT](LICENSE) © Kadir Aksoy