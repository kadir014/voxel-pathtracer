# voxel-pathtracer
A work-in-progress voxel pathtracer prototype in a sandbox environment.

<img src="https://raw.githubusercontent.com/kadir014/voxel-pathtracer/refs/heads/main/data/cornell.jpg" width=400>



# Features
- Real-time
- Global illumination via Monte Carlo pathtracing
- Voxel world traversal accelerated with DDA
- Filmic tonemapping
- Color grading postprocessing
- Simple PBR-ish material pipeline with texture maps
- Custome skydome
- Progressive rendering with frame accumulation
- Russian Roulette path termination

**Roadmap:**
- Disney BRDF
- Procuderal sky / atmosphere model
- Octree acceleration
- Transparent objects & transmission
- Volumetrics
- Denoising
- Low discrepancy noise method 
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
- [Physically Based Rendering Book](https://www.pbr-book.org/4ed/contents)
- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- [Scratchapixel ray tracing articles](https://www.scratchapixel.com/)
- [Free bluenoise textures repository](https://github.com/Calinou/free-blue-noise-textures)
- My previous experiments:
  - [Toy Pathtracer](https://github.com/kadir014/toy-pathtracer)
  - [Radiance Cascades experiments](https://github.com/kadir014/radiance-cascades-experiments)



# License
[MIT](LICENSE) © Kadir Aksoy