"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

from typing import TextIO

from array import array
from struct import unpack, pack
from time import perf_counter
from math import log, sin, cos, radians

import pygame
import moderngl
import numpy

from src import shared
from src.world import BLOCK_IDS
from src.common import MAX_RAYS_PER_PIXEL, OGL_VERSION


start = perf_counter()
print("Loading Heitz Bluenoise data, this may take a little while (I'm going to optimize this I promise 😭)")
# TODO: OPTIMIZE!
import src.heitz as heitz
elapsed = perf_counter() - start
print(f"Loaded Heitz data in {round(elapsed, 3)}s ({round(elapsed*1000.0,3)}ms)")

# Verify heitz data
# TODO: Proper logging...

if len(heitz.RANKING_1SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_2SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_4SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_8SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_16SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_32SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_64SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_128SPP) != 128 * 128 * 8 or \
   len(heitz.RANKING_256SPP) != 128 * 128 * 8:
    print("[ERROR] Heitz ranking data is not correct size!")

if len(heitz.SCRAMBLING_1SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_2SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_4SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_8SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_16SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_32SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_64SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_128SPP) != 128 * 128 * 8 or \
   len(heitz.SCRAMBLING_256SPP) != 128 * 128 * 8:
    print("[ERROR] Heitz scrambling data is not correct size!")

if len(heitz.SOBOL_256SPP_256D) != 256 * 256:
    print("[ERROR] Heitz sobol data is not correct size!")


class RendererSettings:
    def __init__(self, renderer: "Renderer") -> None:
        self.__renderer = renderer

        self.highquality_ray_count = 512
        self.highquality_bounces = 15

        # Only use power of 2s for samples (ray count per pixel)
        self.only_power_of_2s = True
        
        self.denoisers = [
            "None",
            "Bilateral",
        ]

        # 0 -> Pathtraced global illumination
        # 1 -> Normals
        # 2 -> Bounces
        self.pathtracer_output = 0

        self.collect_information = False

        self.sun_azimuth = 0.0
        self.sun_altitude = 30.0

    @property
    def postprocessing(self) -> bool:
        return self.__renderer._post_program["u_enable_post"].value
    
    @postprocessing.setter
    def postprocessing(self, value: bool) -> None:
        self.__renderer._post_program["u_enable_post"].value = value

    @property
    def chromatic_aberration(self) -> float:
        return self.__renderer._post_program["u_chromatic_aberration"].value
    
    @chromatic_aberration.setter
    def chromatic_aberration(self, value: float) -> None:
        self.__renderer._post_program["u_chromatic_aberration"].value = value

    @property
    def tonemapper(self) -> int:
        return self.__renderer._post_program["u_tonemapper"].value
    
    @tonemapper.setter
    def tonemapper(self, value: int) -> None:
        self.__renderer._post_program["u_tonemapper"].value = value

    @property
    def exposure(self) -> float:
        return self.__renderer._post_program["u_exposure"].value
    
    @exposure.setter
    def exposure(self, value: float) -> None:
        self.__renderer._post_program["u_exposure"].value = value

    @property
    def enable_eye_adaptation(self) -> bool:
        data = self.__renderer._exposure_layout_buf.read(4, 4 * 0)
        return unpack("I", data)[0]
    
    @enable_eye_adaptation.setter
    def enable_eye_adaptation(self, value: bool) -> None:
        data = pack("I", value)
        self.__renderer._exposure_layout_buf.write(data, 0)

    @property
    def adapted_exposure(self) -> float:
        data = self.__renderer._exposure_layout_buf.read(4, 4 * 2)
        return unpack("f", data)[0]
    
    @property
    def adaptation_speed(self) -> float:
        data = self.__renderer._exposure_layout_buf.read(4, 4 * 1)
        return unpack("f", data)[0]

    @property
    def brightness(self) -> float:
        return self.__renderer._post_program["u_brightness"].value
    
    @brightness.setter
    def brightness(self, value: float) -> None:
        self.__renderer._post_program["u_brightness"].value = value

    @property
    def contrast(self) -> float:
        return self.__renderer._post_program["u_contrast"].value
    
    @contrast.setter
    def contrast(self, value: float) -> None:
        self.__renderer._post_program["u_contrast"].value = value

    @property
    def saturation(self) -> float:
        return self.__renderer._post_program["u_saturation"].value
    
    @saturation.setter
    def saturation(self, value: float) -> None:
        self.__renderer._post_program["u_saturation"].value = value

    @property
    def ray_count(self) -> int:
        return self.__renderer._pt_program["u_ray_count"].value
    
    @ray_count.setter
    def ray_count(self, value: int) -> None:
        if value < 1:
            return
        
        if value > MAX_RAYS_PER_PIXEL:
            return

        if self.only_power_of_2s:
            nearest_power_of_2 = round(log(value, 2))
            value = int(2.0 ** nearest_power_of_2)

        self.__renderer._pt_program["u_ray_count"].value = value

        #if self.noise_method == 2:
        self.__renderer.write_heinz_data()

    @property
    def bounces(self) -> int:
        return self.__renderer._pt_program["u_bounces"].value
    
    @bounces.setter
    def bounces(self, value: int) -> None:
        self.__renderer._pt_program["u_bounces"].value = value

    @property
    def noise_method(self) -> int:
        return self.__renderer._pt_program["u_noise_method"].value
    
    @noise_method.setter
    def noise_method(self, value: int) -> None:
        self.__renderer._pt_program["u_noise_method"].value = value

    @property
    def russian_roulette(self) -> bool:
        return self.__renderer._pt_program["u_enable_roulette"].value
    
    @russian_roulette.setter
    def russian_roulette(self, value: bool) -> None:
        self.__renderer._pt_program["u_enable_roulette"].value = value

    @property
    def nee(self) -> bool:
        return self.__renderer._pt_program["u_enable_nee"].value
    
    @nee.setter
    def nee(self, value: bool) -> None:
        self.__renderer._pt_program["u_enable_nee"].value = value

    @property
    def enable_sky_texture(self) -> bool:
        return self.__renderer._pt_program["u_enable_sky_texture"].value
    
    @enable_sky_texture.setter
    def enable_sky_texture(self, value: bool) -> None:
        self.__renderer._pt_program["u_enable_sky_texture"].value = value

    @property
    def enable_accumulation(self) -> bool:
        return self.__renderer._pt_program["u_enable_accumulation"].value
    
    @enable_accumulation.setter
    def enable_accumulation(self, value: bool) -> None:
        if self.__renderer._pt_program["u_enable_accumulation"].value != value:
            self.__renderer.clear_accumulation()

        self.__renderer._pt_program["u_enable_accumulation"].value = value

    @property
    def antialiasing(self) -> int:
        return self.__renderer._pt_program["u_antialiasing"].value
    
    @antialiasing.setter
    def antialiasing(self, value: int) -> None:
        self.__renderer._pt_program["u_antialiasing"].value = value

    @property
    def sky_turbidity(self) -> float:
        return self.__renderer._pt_program["u_turbidity"].value
    
    @sky_turbidity.setter
    def sky_turbidity(self, value: float) -> None:
        self.__renderer._pt_program["u_turbidity"].value = value

    @property
    def sun_radiance(self) -> tuple[float, float, float]:
        return self.__renderer._pt_program["u_sun_radiance"].value
    
    @sun_radiance.setter
    def sun_radiance(self, value: tuple[float, float, float]) -> None:
        self.__renderer._pt_program["u_sun_radiance"].value = value

    @property
    def sun_angular_radius(self) -> float:
        return self.__renderer._pt_program["u_sun_angular_radius"].value
    
    @sun_angular_radius.setter
    def sun_angular_radius(self, value: float) -> None:
        self.__renderer._pt_program["u_sun_angular_radius"].value = value

    @property
    def denoiser_id(self) -> int:
        return self.__renderer._denoise_program["u_denoiser"].value
    
    @denoiser_id.setter
    def denoiser_id(self, value: int) -> None:
        self.__renderer._denoise_program["u_denoiser"].value = value

    @property
    def upscaling_method(self) -> int:
        return self.__renderer._upscale_program["u_upscaling_method"].value
    
    @upscaling_method.setter
    def upscaling_method(self, value: int) -> None:
        self.__renderer._upscale_program["u_upscaling_method"].value = value

    def increase_ray_counts(self) -> None:
        """ Increase rays/pixel by one step. """

        if not self.only_power_of_2s:
            self.ray_count += 1

        else:
            power = log(self.ray_count, 2)
            power += 1
            self.ray_count = 2.0 ** power

    def decrease_ray_counts(self) -> None:
        """ Decrease rays/pixel by one step. """

        if not self.only_power_of_2s:
            self.ray_count -= 1

        else:
            power = log(self.ray_count, 2)
            power -= 1
            self.ray_count = 2.0 ** power
    
    def color_profile_custom(self) -> None:
        """ Set to custom color profile which I thought looked good with ACES. """

        self.exposure = -2.1
        self.brightness = -0.0052
        self.contrast = 1.206
        self.saturation = 1.32

    def color_profile_zero(self) -> None:
        """ Set color grading to defaults. """

        self.exposure = 0.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0

    def color_profile_noir(self) -> None:
        self.exposure = -3.6
        self.brightness = -0.056
        self.contrast = 1.556
        self.saturation = 0.0


class ShaderPatcher:
    def __init__(self) -> None:
        self.include_pattern = "#include"

        self.headers: dict[str, str] = {}
        self.shaders: dict[str, str] = {}

    def patch_stream(self, stream: TextIO) -> str:
        new_content = ""

        omit_start = False

        for line in stream.read().split("\n"):
            line = line.strip()

            if line.startswith("/* OMIT END */"):
                omit_start = False
                continue

            if line.startswith("/* OMIT START */"):
                omit_start = True

            if omit_start:
                continue

            if line.startswith(self.include_pattern):
                path = " ".join(line.split(" ")[1:])

                # Remove quotes
                if path.startswith("\"") and path.startswith("\""):
                    path = path[1:-1]

                # If header is already patched, get from cache
                # else, patch it
                if path in self.headers:
                    new_content += self.headers[path]

                else:
                    path = "src/shaders/" + path

                    with open(path, "r", encoding="utf-8") as f:
                        include_content = self.patch_stream(f)

                    new_content += include_content

            else:
                new_content += line + "\n"

        return new_content
    
    def patch_header(self, name: str, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            self.headers[name] = self.patch_stream(f)

    def patch_file(self, name: str, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            self.shaders[name] = self.patch_stream(f)


class Renderer:
    """
    Pathtraced rendering engine.
    """

    def __init__(self,
            resolution: tuple[int, int],
            logical_resolution: tuple[int, int]
            ) -> None:
        self._resolution = resolution
        self._logical_resolution = logical_resolution

        ogl_version = int(f"{OGL_VERSION[0]}{OGL_VERSION[1]}0")
        self._context = moderngl.create_context(require=ogl_version)

        # Everything is rendered on a screenquad
        base_vertex_shader = """
        #version 330

        in vec2 in_position;
        in vec2 in_uv;

        out vec2 v_uv;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);

            v_uv = in_uv;
        }
        """

        self.patcher = ShaderPatcher()
        self.patcher.patch_header("common.glsl", "src/shaders/common.glsl")
        self.patcher.patch_header("color.glsl", "src/shaders/color.glsl")
        self.patcher.patch_header("microfacet.glsl", "src/shaders/microfacet.glsl")
        self.patcher.patch_header("bicubic.glsl", "src/shaders/bicubic.glsl")
        self.patcher.patch_header("fxaa.glsl", "src/shaders/fxaa.glsl")
        self.patcher.patch_header("preetham.glsl", "src/shaders/preetham.glsl")
        self.patcher.patch_file("post.fsh", "src/shaders/post.fsh")
        self.patcher.patch_file("upscale.fsh", "src/shaders/upscale.fsh")
        self.patcher.patch_file("pathtracer.fsh", "src/shaders/pathtracer.fsh")
        self.patcher.patch_file("fxaa_pass.fsh", "src/shaders/fxaa_pass.fsh")
        self.patcher.patch_file("denoise.fsh", "src/shaders/denoise.fsh")

        # All VAOs will use the same buffers since they are all just plain screen quads
        self._vbo = self.create_buffer_object([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        self._uvbo = self.create_buffer_object([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self._ibo = self.create_buffer_object([0, 1, 2, 1, 2, 3])

        self._post_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=self.patcher.shaders["post.fsh"]
        )
        self._post_program["u_enable_post"] = True
        self._post_program["u_tonemapper"] = 1
        self._post_program["u_chromatic_aberration"] = 0.0035
        self._post_program["s_texture"] = 0
        self._post_program["s_lum"] = 1

        self._post_vao = self._context.vertex_array(
            self._post_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._pt_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=self.patcher.shaders["pathtracer.fsh"]
        )
        self._pt_program["s_grid"] = 0
        self._pt_program["s_sky"] = 1
        self._pt_program["s_albedo_atlas"] = 2
        self._pt_program["s_emissive_atlas"] = 3
        self._pt_program["s_roughness_atlas"] = 4
        self._pt_program["s_metallic_atlas"] = 5
        self._pt_program["s_glass_atlas"] = 6
        self._pt_program["s_previous_frame"] = 7
        self._pt_program["s_previous_normal"] = 8
        self._pt_program["u_ray_count"] = 8
        self._pt_program["u_bounces"] = 3
        self._pt_program["u_noise_method"] = 1
        self._pt_program["u_enable_roulette"] = False
        self._pt_program["u_enable_sky_texture"] = False
        self._pt_program["u_enable_nee"] = True
        self._pt_program["u_resolution"] = self._logical_resolution
        self._pt_program["u_voxel_size"] = shared.world.voxel_size
        self._pt_program["u_enable_accumulation"] = True
        self._pt_program["u_antialiasing"] = 2
        self._pt_program["u_exp_raymarch"] = 0
        self._pt_program["u_turbidity"] = 2.619
        self._pt_program["u_sun_direction"] = pygame.Vector3(0.0, 1.0, 1.0).normalize()
        self._pt_program["u_sun_radiance"] = (1500.0, 1500.0, 1500.0)
        self._pt_program["u_sun_angular_radius"] = 0.0275

        self._pt_vao = self._context.vertex_array(
            self._pt_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._upscale_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=self.patcher.shaders["upscale.fsh"]
        )
        self._upscale_program["s_texture"] = 0
        self._upscale_program["s_overlay"] = 1
        self._upscale_program["u_resolution"] = self._logical_resolution
        self._upscale_program["u_upscaling_method"] = 2

        self._upscale_vao = self._context.vertex_array(
            self._upscale_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._fxaa_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=self.patcher.shaders["fxaa_pass.fsh"]
        )
        self._fxaa_program["s_texture"] = 0
        self._fxaa_program["u_resolution"] = self._logical_resolution

        self._fxaa_vao = self._context.vertex_array(
            self._fxaa_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._denoise_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=self.patcher.shaders["denoise.fsh"]
        )
        self._denoise_program["s_texture"] = 0
        self._denoise_program["u_resolution"] = self._logical_resolution
        self._denoise_program["u_denoiser"] = 0
        self._denoise_program["u_hw"] = 4
        self._denoise_program["u_sigmaspace"] = 10.0
        self._denoise_program["u_sigmacolor"] = 25.0

        self._denoise_vao = self._context.vertex_array(
            self._denoise_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        self._pathtracer_target_normal0 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_normal0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pathtracer_target_normal1 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_normal1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pathtracer_target_bounces0 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_bounces0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pathtracer_target_bounces1 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_bounces1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Note: data type being f4 lets us store colors in HDR
        # Alpha channel in pathtracing textures is for luminance
        self._pathtracer_target_texture0 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_texture0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pathtracer_target_texture0.repeat_x = False
        self._pathtracer_target_texture0.repeat_y = False
        self._pathtracer_fbo0 = self._context.framebuffer(color_attachments=(
            self._pathtracer_target_texture0,self._pathtracer_target_normal0, self._pathtracer_target_bounces0)
        )
        self._pathtracer_target_texture1 = self._context.texture(self._logical_resolution, 4, dtype="f4")
        self._pathtracer_target_texture1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._pathtracer_target_texture1.repeat_x = False
        self._pathtracer_target_texture1.repeat_y = False
        self._pathtracer_fbo1 = self._context.framebuffer(color_attachments=(
            self._pathtracer_target_texture1,self._pathtracer_target_normal1, self._pathtracer_target_bounces1)
        )


        self._fxaa_target_texture = self._context.texture(self._logical_resolution, 3, dtype="f1")
        self._fxaa_target_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._fxaa_fbo = self._context.framebuffer(color_attachments=(self._fxaa_target_texture,))

        self._post_target_texture = self._context.texture(self._logical_resolution, 3, dtype="f1")
        self._post_target_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._post_fbo = self._context.framebuffer(color_attachments=(self._post_target_texture,))

        self._denoise_target_texture = self._context.texture(self._logical_resolution, 3, dtype="f4")
        self._denoise_target_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._denoise_target_texture.repeat_x = False
        self._denoise_target_texture.repeat_y = False
        self._denoise_fbo = self._context.framebuffer(color_attachments=(self._denoise_target_texture,))


        # The post-processing shader will rewrite the adapted exposure each frame
        # so make the buffer dynamic
        exposure_total_size = 4 * 4 * 4
        self._exposure_layout_buf = self._context.buffer(reserve=exposure_total_size, dynamic=True)
        self._exposure_layout_buf.write(pack("Iff", 0, 0.001, 10.0), 0)
        self._exposure_layout_buf.bind_to_storage_buffer(1)


        acc_total_size = self._logical_resolution[0] * self._logical_resolution[1] * 4
        self._acc_layout_buf = self._context.buffer(reserve=acc_total_size, dynamic=True)
        self._acc_layout_buf.bind_to_storage_buffer(2)
        self.clear_accumulation()


        self.settings = RendererSettings(self)
        self.settings.color_profile_custom()


        # We gotta merge all members of the SSBO layout!
        heinz_total_size = 128 * 128 * 8 * 4 + 128 * 128 * 8 * 4 + 256 * 256 * 4
        # spp is not changed every frame, we don't need dynamic buffers
        self.heinz_layout_buf = self._context.buffer(reserve=heinz_total_size, dynamic=False)
        self.heinz_layout_buf.bind_to_storage_buffer(0)

        self.write_heinz_data()


        sky_surf = pygame.image.load("data/sky/voortrekker_interior.png")
        self._sky_texture = self._context.texture(
            sky_surf.get_size(),
            3,
            pygame.image.tobytes(sky_surf, "RGB", True)
        )
        self._sky_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._sky_texture.repeat_x = True
        self._sky_texture.repeat_y = True

        self._voxel_tex = self._context.texture3d(shared.world.dimensions, 1)
        self._voxel_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)


        self.block_texture_size = (16, 16)

        self.scalar_textures = {}

        self.block_textures = {
            "cobblestone": pygame.image.load("data/blocks/cobblestone.png"),
            "dirt": pygame.image.load("data/blocks/dirt.png"),
            "glowstone": pygame.image.load("data/blocks/glowstone.png"),
            "glowstone_emissive": pygame.image.load("data/blocks/glowstone.png"),
            "grass_top": pygame.image.load("data/blocks/grass_top.png"),
            "grass_side": pygame.image.load("data/blocks/grass_side.png"),
            "iron_block": pygame.image.load("data/blocks/iron_block.png"),
            "iron_block_roughness": pygame.image.load("data/blocks/iron_block_roughness.png"),
            "lime_wool": pygame.image.load("data/blocks/lime_wool.png"),
            "red_wool": pygame.image.load("data/blocks/red_wool.png"),
            "red_light_emissive": pygame.image.load("data/blocks/red_light_emissive.png"),
            "copper_block": pygame.image.load("data/blocks/copper_block.png"),
            "copper_block_roughness": pygame.image.load("data/blocks/copper_block_roughness.png")
        }

        # albedo atlas
        self.block_atlas = {
            # top bottom side
            "cobblestone": ("cobblestone", "cobblestone", "cobblestone"),
            "dirt": ("dirt", "dirt", "dirt"),
            "glowstone": ("glowstone", "glowstone", "glowstone"),
            "grass": ("grass_top", "dirt", "grass_side"),
            "iron_block": ("iron_block", "iron_block", "iron_block"),
            "lime_wool": ("lime_wool", "lime_wool", "lime_wool"),
            "red_wool": ("red_wool", "red_wool", "red_wool"),
            "red_light": (0, 0, 0),
            "copper_block": ("copper_block", "copper_block", "copper_block")
        }

        self.emissive_atlas = {
            # top bottom side
            "cobblestone":(0, 0, 0),
            "dirt": (0, 0, 0),
            "glowstone": ("glowstone_emissive", "glowstone_emissive", "glowstone_emissive"),
            "grass": (0, 0, 0),
            "iron_block": (0, 0, 0),
            "lime_wool": (0, 0, 0),
            "red_wool": (0, 0, 0),
            "red_light": ("red_light_emissive", "red_light_emissive", "red_light_emissive"),
            "copper_block": (0, 0, 0),
        }

        self.roughness_atlas = {
            # top bottom side
            "cobblestone": (50, 50, 50),
            "dirt": (50, 50, 50),
            "glowstone": (50, 50, 50),
            "grass": (50, 50, 50),
            "iron_block": ("iron_block_roughness", "iron_block_roughness", "iron_block_roughness"),
            "lime_wool": (50, 50, 50),
            "red_wool": (0, 0, 0),
            "red_light": (50, 50, 50),
            "copper_block": ("copper_block_roughness", "copper_block_roughness", "copper_block_roughness")
        }

        self.metallic_atlas = {
            # top bottom side
            "cobblestone": (0, 0, 0),
            "dirt": (0, 0, 0),
            "glowstone": (0, 0, 0),
            "grass": (0, 0, 0),
            "iron_block": (100, 100, 100),
            "lime_wool": (0, 0, 0),
            "red_wool": (0, 0, 0),
            "red_light": (0, 0, 0),
            "copper_block": (100, 100, 100)
        }

        self.glass_atlas = {
            # top bottom side
            "cobblestone": (0, 0, 0),
            "dirt": (0, 0, 0),
            "glowstone": (0, 0, 0),
            "grass": (0, 0, 0),
            "iron_block": (0, 0, 0),
            "lime_wool": (0, 0, 0),
            "red_wool": (100, 100, 100),
            "red_light": (0, 0, 0),
            "copper_block": (0, 0, 0)
        }

        # Cache scalar surfaces
        for i in range(101):
            s = pygame.Surface(self.block_texture_size)
            c = int(i / 100.0 * 255.0)
            s.fill((c, c, c))
            self.block_textures[i] = s

        self.generate_block_atlas_texture()
        self.generate_emissive_atlas_texture()
        self.generate_roughness_atlas_texture()
        self.generate_metallic_atlas_texture()
        self.generate_glass_atlas_texture()


        self.ui_surface = pygame.Surface(self._resolution, pygame.SRCALPHA)
        self.ui_surface.fill((0, 0, 0, 0))
        self.ui_texture = self._context.texture(self._resolution, 4)
        self.null_ui_texture = self._context.texture(self._resolution, 4)

        self.ui_scale = 2.0
        self.crosshair_surf = pygame.image.load("data/ui/crosshair.png")
        self.hotbar_surf = pygame.transform.scale_by(pygame.image.load("data/ui/hotbar.png"), self.ui_scale)
        self.hotbar_sel_surf = pygame.transform.scale_by(pygame.image.load("data/ui/hotbar_selection.png"), self.ui_scale)

        self.update_ui()

        self.pingpong_frame = 0

        # Total number of rays shot from camera and bounces this frame
        self.frame_ray_count = 0

    def __del__(self) -> None:
        self._context.release()

    def create_buffer_object(self, data: list) -> moderngl.Buffer:
        """ Create buffer object from array. """

        dtype = "f" if isinstance(data[0], float) else "I"
        return self._context.buffer(array(dtype, data))
    
    def write_heinz_data(self) -> None:
        start = perf_counter()

        spp = self.settings.ray_count

        scrambling_map = {
            1: heitz.SCRAMBLING_1SPP,
            2: heitz.SCRAMBLING_2SPP,
            4: heitz.SCRAMBLING_4SPP,
            8: heitz.SCRAMBLING_8SPP,
            16: heitz.SCRAMBLING_16SPP,
            32: heitz.SCRAMBLING_32SPP,
            64: heitz.SCRAMBLING_64SPP,
            128: heitz.SCRAMBLING_128SPP,
            256: heitz.SCRAMBLING_256SPP,
        }

        ranking_map = {
            1: heitz.RANKING_1SPP,
            2: heitz.RANKING_2SPP,
            4: heitz.RANKING_4SPP,
            8: heitz.RANKING_8SPP,
            16: heitz.RANKING_16SPP,
            32: heitz.RANKING_32SPP,
            64: heitz.RANKING_64SPP,
            128: heitz.RANKING_128SPP,
            256: heitz.RANKING_256SPP,
        }

        if spp not in scrambling_map:
            old_spp = spp
            nearest_power_of_2 = round(log(spp, 2))
            spp = int(2.0 ** nearest_power_of_2)

            spp = min(spp, 256)

            print(f"[WARNING] Heitz data is not optimized for {old_spp}spp, results may be unexpected or non-optimal. Changed spp = {spp}")

        scrambling = scrambling_map[spp]
        ranking = ranking_map[spp]

        buf_data = array("I", ranking + scrambling + heitz.SOBOL_256SPP_256D)
        self.heinz_layout_buf.write(buf_data)

        elapsed = perf_counter() - start
        print(f"Wrote Heitz data for {spp}spp onto GPU in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")

    def clear_accumulation(self) -> None:
        self._acc_layout_buf.clear()
    
    def update_ui_surface(self, hotbar: int = 0) -> None:
        self.ui_surface.fill((0, 0, 0, 0))

        crosshair_pos = (self._resolution[0]*0.5, self._resolution[1]*0.5)
        self.ui_surface.blit(self.crosshair_surf, self.crosshair_surf.get_rect(center=crosshair_pos))

        hotbar_pos = (self._resolution[0]*0.5, self._resolution[1] - self.hotbar_surf.height)
        self.ui_surface.blit(self.hotbar_surf, self.hotbar_surf.get_rect(centerx=hotbar_pos[0], top=hotbar_pos[1]))

        hotbar_single_size = self.hotbar_surf.width / 9

        hotbar_sel_pos = (
            self._resolution[0]*0.5 - self.hotbar_surf.width * 0.5 + hotbar_single_size * hotbar + hotbar_single_size * 0.5,
            self._resolution[1] - self.hotbar_surf.height
        )
        self.ui_surface.blit(self.hotbar_sel_surf, self.hotbar_sel_surf.get_rect(centerx=hotbar_sel_pos[0], top=hotbar_sel_pos[1]))

        for i in range(1, min(9, len(BLOCK_IDS)) + 1):
            block = BLOCK_IDS[i]

            if block == "grass":
                block = "grass_side"

            if block == "red_light":
                block = "red_light_emissive"

            block_surf = pygame.transform.scale_by(self.block_textures[block], 1.4)

            x = self._resolution[0] * 0.5 - self.hotbar_surf.width * 0.495 - hotbar_single_size * 0.5
            x += i * hotbar_single_size
            y = self._resolution[1] - self.hotbar_surf.height + hotbar_single_size * 0.55
            self.ui_surface.blit(block_surf, block_surf.get_frect(center=(x,y)))
    
    def update_ui_texture(self) -> None:
        self.ui_texture.write(pygame.image.tobytes(self.ui_surface, "RGBA", True))

    def update_ui(self, hotbar: int = 0) -> None:
        start = perf_counter()
        self.update_ui_surface(hotbar)
        self.update_ui_texture()
        elapsed = perf_counter() - start
        print(f"Updated UI in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")

    def update_grid_texture(self) -> None:
        start = perf_counter()

        layers = []
        for layer_i in range(shared.world.dimensions[1]):
            surf = pygame.Surface((shared.world.dimensions[0], shared.world.dimensions[2]))
            layers.append(surf)

            # z -> layer_i
            # x -> x
            # y = height - y

            for y in range(shared.world.dimensions[1]):
                for x in range(shared.world.dimensions[0]):
                    voxel = shared.world[x, y, layer_i]

                    surf.set_at((x, (shared.world.dimensions[1] - 1) - y), (voxel, voxel, voxel))

        data = bytearray()
        for layer in layers:
            layer_data = pygame.image.tobytes(layer, "RGB", True)
            for i, byte in enumerate(layer_data):
                # Only the red channel
                if (i % 3 == 0):
                    data.append(byte)

        self._voxel_tex.write(data)

        elapsed = perf_counter() - start
        print(f"Updated world texture in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")

    def generate_block_atlas_texture(self) -> None:
        # atlas -> 3xN
        # N is the amount of block variants
        # each block texture is 16x16

        n_blocks = len(self.block_atlas)

        self.block_atlas_tex = self._context.texture(
            (self.block_texture_size[0] * 3, self.block_texture_size[1] * n_blocks),
            3
        )
        self.block_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Block atlas size: {self.block_atlas_tex.width}x{self.block_atlas_tex.height}")

        for i, block_variant in enumerate(BLOCK_IDS):
            if block_variant is None: continue

            top_surf = self.block_textures[self.block_atlas[block_variant][0]]
            bottom_surf = self.block_textures[self.block_atlas[block_variant][1]]
            side_surf = self.block_textures[self.block_atlas[block_variant][2]]
            top_data = pygame.image.tobytes(top_surf, "RGB", True)
            bottom_data = pygame.image.tobytes(bottom_surf, "RGB", True)
            side_data = pygame.image.tobytes(side_surf, "RGB", True)

            self.block_atlas_tex.write(
                top_data,
                viewport=(
                    self.block_texture_size[0] * 0,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.block_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_texture_size[0] * 1,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.block_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_texture_size[0] * 2,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
    
    def generate_emissive_atlas_texture(self) -> None:
        n_blocks = len(self.block_atlas)

        self.emissive_atlas_tex = self._context.texture(
            (self.block_texture_size[0] * 3, self.block_texture_size[1] * n_blocks),
            3
        )
        self.emissive_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Emissive atlas size: {self.emissive_atlas_tex.width}x{self.emissive_atlas_tex.height}")

        for i, block_variant in enumerate(BLOCK_IDS):
            if block_variant is None: continue

            top_surf = self.block_textures[self.emissive_atlas[block_variant][0]]
            bottom_surf = self.block_textures[self.emissive_atlas[block_variant][1]]
            side_surf = self.block_textures[self.emissive_atlas[block_variant][2]]
            top_data = pygame.image.tobytes(top_surf, "RGB", True)
            bottom_data = pygame.image.tobytes(bottom_surf, "RGB", True)
            side_data = pygame.image.tobytes(side_surf, "RGB", True)

            self.emissive_atlas_tex.write(
                top_data,
                viewport=(
                    self.block_texture_size[0] * 0,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.emissive_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_texture_size[0] * 1,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.emissive_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_texture_size[0] * 2,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )

    def generate_roughness_atlas_texture(self) -> None:
        n_blocks = len(self.block_atlas)

        self.roughness_atlas_tex = self._context.texture(
            (self.block_texture_size[0] * 3, self.block_texture_size[1] * n_blocks),
            3
        )
        self.roughness_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Roughness atlas size: {self.roughness_atlas_tex.width}x{self.roughness_atlas_tex.height}")

        for i, block_variant in enumerate(BLOCK_IDS):
            if block_variant is None: continue

            top_surf = self.block_textures[self.roughness_atlas[block_variant][0]]
            bottom_surf = self.block_textures[self.roughness_atlas[block_variant][1]]
            side_surf = self.block_textures[self.roughness_atlas[block_variant][2]]
            top_data = pygame.image.tobytes(top_surf, "RGB", True)
            bottom_data = pygame.image.tobytes(bottom_surf, "RGB", True)
            side_data = pygame.image.tobytes(side_surf, "RGB", True)

            self.roughness_atlas_tex.write(
                top_data,
                viewport=(
                    self.block_texture_size[0] * 0,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.roughness_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_texture_size[0] * 1,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.roughness_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_texture_size[0] * 2,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )

    def generate_metallic_atlas_texture(self) -> None:
        n_blocks = len(self.block_atlas)

        self.metallic_atlas_tex = self._context.texture(
            (self.block_texture_size[0] * 3, self.block_texture_size[1] * n_blocks),
            3
        )
        self.metallic_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Metallic atlas size: {self.metallic_atlas_tex.width}x{self.metallic_atlas_tex.height}")

        for i, block_variant in enumerate(BLOCK_IDS):
            if block_variant is None: continue

            top_surf = self.block_textures[self.metallic_atlas[block_variant][0]]
            bottom_surf = self.block_textures[self.metallic_atlas[block_variant][1]]
            side_surf = self.block_textures[self.metallic_atlas[block_variant][2]]
            top_data = pygame.image.tobytes(top_surf, "RGB", True)
            bottom_data = pygame.image.tobytes(bottom_surf, "RGB", True)
            side_data = pygame.image.tobytes(side_surf, "RGB", True)

            self.metallic_atlas_tex.write(
                top_data,
                viewport=(
                    self.block_texture_size[0] * 0,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.metallic_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_texture_size[0] * 1,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.metallic_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_texture_size[0] * 2,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )

    def generate_glass_atlas_texture(self) -> None:
        n_blocks = len(self.block_atlas)

        self.glass_atlas_tex = self._context.texture(
            (self.block_texture_size[0] * 3, self.block_texture_size[1] * n_blocks),
            3
        )
        self.glass_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Glass atlas size: {self.glass_atlas_tex.width}x{self.glass_atlas_tex.height}")

        for i, block_variant in enumerate(BLOCK_IDS):
            if block_variant is None: continue

            top_surf = self.block_textures[self.glass_atlas[block_variant][0]]
            bottom_surf = self.block_textures[self.glass_atlas[block_variant][1]]
            side_surf = self.block_textures[self.glass_atlas[block_variant][2]]
            top_data = pygame.image.tobytes(top_surf, "RGB", True)
            bottom_data = pygame.image.tobytes(bottom_surf, "RGB", True)
            side_data = pygame.image.tobytes(side_surf, "RGB", True)

            self.glass_atlas_tex.write(
                top_data,
                viewport=(
                    self.block_texture_size[0] * 0,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.glass_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_texture_size[0] * 1,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )
            self.glass_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_texture_size[0] * 2,
                    self.block_texture_size[1] * (i-1),
                    self.block_texture_size[0],
                    self.block_texture_size[1]
                )
            )

    def get_frame_ray_count(self, pt_fbo: moderngl.Framebuffer) -> None:
        # This calculation is probably wrong, I need to rewrite it

        frame_bounces = 0

        # f1 -> unsigned byte
        buffer = pt_fbo.read(attachment=2, components=1, dtype="f1")
        data = numpy.frombuffer(buffer, dtype=numpy.uint8)
        data = data.reshape((self._logical_resolution[0], self._logical_resolution[1], 1))

        nee_factor = 1.0
        if self.settings.nee:
            nee_factor = 2.0

        max_bounces = float(self.settings.bounces)
        max_samples = float(self.settings.ray_count)
        inv_samples = 1.0 / 255.0

        camera_rays = numpy.sum(data + 1)

        # go back to total_bounces
        # float normalized_bounces = (float(total_bounces) / float(u_ray_count) / float(u_bounces));
        pixel_bounces = data * inv_samples * max_bounces * max_samples * nee_factor
        world_bounces = numpy.sum(pixel_bounces)

        frame_bounces = camera_rays * max_samples + world_bounces

        # Sample count x bounces
        self.frame_ray_count = frame_bounces

    def render(self, ui: bool = True) -> None:
        """
        Render one frame.
        
        Render passes:
        --------------

          Pathtracing
              ↓           (HDR, logical resolution, NEAREST)
          Denoising
              ↓
        Post-processing
              ↓           (LDR, logical resolution, LINEAR)
             FXAA
              ↓ 
          Upscaling
              ↓           (LDR, display resolution, LINEAR)
          UI overlay
              ↓
           Display
        """

        # Build sun direction from spherical coords
        # Altitude is the same as "pitch"
        # Azimuth is the horizontal angle of the sun measured clockwise from North
        # it's also almost the same as "yaw"

        pitch_r = radians(self.settings.sun_altitude)
        yaw_r = radians(self.settings.sun_azimuth)
        pitch_c = cos(pitch_r)
        pitch_s = sin(pitch_r)
        yaw_c = cos(yaw_r)
        yaw_s = sin(yaw_r)

        sun_dir = pygame.Vector3(
            yaw_c * pitch_c,
            pitch_s,
            yaw_s * pitch_c
        ).normalize()

        self._pt_program["u_sun_direction"] = tuple(sun_dir)

        self._context.disable(moderngl.BLEND)

        targets = (self._pathtracer_target_texture0, self._pathtracer_target_texture1)
        fbos = (self._pathtracer_fbo0, self._pathtracer_fbo1)
        normals = (self._pathtracer_target_normal0, self._pathtracer_target_normal1)
        bounces = (self._pathtracer_target_bounces0, self._pathtracer_target_bounces1)

        # frame 0:
        # current fbo -> 0
        # current target -> 0
        # previous target -> 1
        #
        # frame 1:
        # current fbo -> 1
        # current target -> 1
        # previous target -> 0

        frame = self.pingpong_frame
        current_fbo = fbos[frame % 2]
        current_target = targets[frame % 2]
        previous_target = targets[(frame + 1 ) % 2]
        current_normal = normals[frame % 2]
        previous_normal = normals[(frame + 1 ) % 2]
        current_bounces = bounces[frame % 2]

        current_fbo.use()
        self._voxel_tex.use(0)
        self._sky_texture.use(1)
        self.block_atlas_tex.use(2)
        self.emissive_atlas_tex.use(3)
        self.roughness_atlas_tex.use(4)
        self.metallic_atlas_tex.use(5)
        self.glass_atlas_tex.use(6)
        previous_target.use(7)
        previous_normal.use(8)
        self._pt_vao.render()

        if (self.settings.enable_eye_adaptation):
            current_target.build_mipmaps()
            current_target.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_LINEAR)
        else:
            current_target.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if self.settings.pathtracer_output == 0:
            ... # Current target stays the same
        elif self.settings.pathtracer_output == 1:
            current_target = current_normal
        elif self.settings.pathtracer_output == 2:
            current_target = current_bounces

        self._denoise_fbo.use()
        current_target.use(0)
        self._denoise_vao.render()

        if self.settings.antialiasing in (0, 1):
            self._post_fbo.use()
            self._denoise_target_texture.use(0)
            current_target.use(1)
            self._post_vao.render()

        elif self.settings.antialiasing == 2:
            self._fxaa_fbo.use()
            self._denoise_target_texture.use(0)
            current_target.use(1)
            self._post_vao.render()

            self._post_fbo.use()
            self._fxaa_target_texture.use(0)
            self._fxaa_vao.render()

        self._context.screen.use()
        self._post_target_texture.use(0)
        if ui:
            self.ui_texture.use(1)
        else:
            self.null_ui_texture.use(1)
        self._upscale_vao.render()

        if self.settings.enable_accumulation:
            self.pingpong_frame += 1
        else:
            self.pingpong_frame = 0
        
        if self.settings.collect_information:
            self.get_frame_ray_count(current_fbo)

    def high_quality_snapshot(self) -> None:
        """ Render with temporary high quality settings and save a snapshot. """
        start = perf_counter()

        old_ray_count = self.settings.ray_count
        old_bounces = self.settings.bounces
        old_enable_accumulation = self.settings.enable_accumulation

        self._pt_program["u_ray_count"].value = self.settings.highquality_ray_count
        self.settings.bounces = self.settings.highquality_bounces
        self.settings.enable_accumulation = False

        # Manually invoke this since we don't interact with RendererSettings property
        #if self.settings.noise_method == 2:
        self.write_heinz_data()

        self.render(ui=False)

        surf = pygame.image.frombytes(
            self._context.screen.read(),
            self._resolution,
            "RGB"
        )
        pygame.image.save(pygame.transform.flip(surf, False, True), "snapshot.png")

        self._pt_program["u_ray_count"].value = old_ray_count
        self.settings.bounces = old_bounces
        self.settings.enable_accumulation = old_enable_accumulation

        elapsed = perf_counter() - start
        print(f"High quality render in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")