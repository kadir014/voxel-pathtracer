"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from array import array

import pygame
import moderngl


class RendererSettings:
    def __init__(self, renderer: "Renderer") -> None:
        self.__renderer = renderer

    @property
    def postprocessing(self) -> bool:
        return self.__renderer._post_program["u_enable_post"].value
    
    @postprocessing.setter
    def postprocessing(self, value: bool) -> None:
        self.__renderer._post_program["u_enable_post"].value = value

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
        self.__renderer._pt_program["u_ray_count"].value = value

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


class Renderer:
    """
    Pathtraced rendering engine.
    """

    def __init__(self,
            resolution: tuple[int, int],
            logical_resolution: tuple[int, int]
            ) -> None:
        self.resolution = resolution
        self.logical_resolution = logical_resolution

        self._context = moderngl.create_context()

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

        # All VAOs will use the same buffers since they are all just plain screen quads
        self._vbo = self.create_buffer_object([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        self._uvbo = self.create_buffer_object([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self._ibo = self.create_buffer_object([0, 1, 2, 1, 2, 3])

        self._post_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=open("src/shaders/post.fsh").read()
        )

        self._post_program["u_enable_post"] = True
        self._post_program["u_tonemapper"] = 1
        self._post_program["u_exposure"] = -1.993
        self._post_program["u_brightness"] = 0.0071
        self._post_program["u_contrast"] = 1.0212
        self._post_program["u_saturation"] = 1.225

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
            fragment_shader=open("src/shaders/pathtracer.fsh").read()
        )

        self._pt_program["s_bluenoise"] = 0
        self._pt_program["s_grid"] = 1
        self._pt_program["s_sky"] = 2
        self._pt_program["s_albedo_atlas"] = 3
        self._pt_program["s_emissive_atlas"] = 4
        self._pt_program["u_ray_count"] = 8
        self._pt_program["u_bounces"] = 2
        self._pt_program["u_noise_method"] = 1
        self._pt_program["u_resolution"] = self.logical_resolution

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
            fragment_shader=open("src/shaders/upscale.fsh").read()
        )

        self._upscale_vao = self._context.vertex_array(
            self._upscale_program,
            (
                (self._vbo, "2f", "in_position"),
                (self._uvbo, "2f", "in_uv")
            ),
            self._ibo
        )

        bluenoise_surf = pygame.image.load("data/bluenoise_1024x1024.png")
        self._bluenoise_texture = self._context.texture(
            bluenoise_surf.get_size(),
            4,
            pygame.image.tobytes(bluenoise_surf, "RGBA", True)
        )
        self._bluenoise_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._bluenoise_texture.repeat_x = True
        self._bluenoise_texture.repeat_y = True

        # Note: data type being f4 lets us store colors in HDR
        self._pathtracer_target_texture = self._context.texture(self.logical_resolution, 3, dtype="f4")
        self._pathtracer_target_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._pathtracer_fbo = self._context.framebuffer(color_attachments=(self._pathtracer_target_texture,))

        self._post_target_texture = self._context.texture(self.logical_resolution, 3, dtype="f1")
        self._post_target_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._post_fbo = self._context.framebuffer(color_attachments=(self._post_target_texture,))

        sky_surf = pygame.image.load("data/qwantani_dusk_2_puresky.png")
        self._sky_texture = self._context.texture(
            sky_surf.get_size(),
            3,
            pygame.image.tobytes(sky_surf, "RGB", True)
        )
        self._sky_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._sky_texture.repeat_x = True
        self._sky_texture.repeat_y = True

        self.grid_size = (10, 10, 10) # x, y, z
        self._voxel_tex = self._context.texture3d(self.grid_size, 1)
        self._voxel_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.grid = {}
        for y in range(self.grid_size[1]):
            for z in range(self.grid_size[2]):
                for x in range(self.grid_size[0]):
                    self.grid[(x, y, z)] = 0

        self.settings = RendererSettings(self)


        self.block_size = (16, 16)

        self.block_order = (
            None, # 0 is air
            "cobblestone",
            "dirt",
            "glowstone",
            "grass"
        )

        self.block_textures = {
            None: pygame.Surface(self.block_size),
            "cobblestone": pygame.image.load("data/blocks/cobblestone.png"),
            "dirt": pygame.image.load("data/blocks/dirt.png"),
            "glowstone": pygame.image.load("data/blocks/glowstone.png"),
            "glowstone_emissive": pygame.image.load("data/blocks/glowstone.png"),
            "grass_top": pygame.image.load("data/blocks/grass_top.png"),
            "grass_side": pygame.image.load("data/blocks/grass_side.png"),
        }

        self.block_atlas = {
            # top bottom side
            "cobblestone": ("cobblestone", "cobblestone", "cobblestone"),
            "dirt": ("dirt", "dirt", "dirt"),
            "glowstone": ("glowstone", "glowstone", "glowstone"),
            "grass": ("grass_top", "dirt", "grass_side")
        }

        self.emissive_atlas = {
            # top bottom side
            "cobblestone": (None, None, None),
            "dirt": (None, None, None),
            "glowstone": ("glowstone_emissive", "glowstone_emissive", "glowstone_emissive"),
            #"glowstone": (None, None, None),
            "grass": (None, None, None)
        }

        self.generate_block_atlas_texture()
        self.generate_emissive_atlas_texture()

    def __del__(self) -> None:
        self._context.release()

    def create_buffer_object(self, data: list) -> moderngl.Buffer:
        """ Create buffer object from array. """

        dtype = "f" if isinstance(data[0], float) else "I"
        return self._context.buffer(array(dtype, data))

    def update_grid(self) -> None:
        layers = []
        for layer_i in range(self.grid_size[1]):
            surf = pygame.Surface((self.grid_size[0], self.grid_size[2]))
            layers.append(surf)

            # z -> layer_i
            # x -> x
            # y = height - y

            for y in range(self.grid_size[1]):
                for x in range(self.grid_size[0]):
                    voxel = self.grid[x, y, layer_i]

                    surf.set_at((x, (self.grid_size[1] - 1) - y), (voxel, voxel, voxel))

        data = bytearray()
        for layer in layers:
            layer_data = pygame.image.tobytes(layer, "RGB", True)
            for i, byte in enumerate(layer_data):
                # Only the red channel
                if (i % 3 == 0):
                    data.append(byte)

        self._voxel_tex.write(data)

    def generate_block_atlas_texture(self) -> None:
        # atlas -> 3xN
        # N is the amount of block variants
        # each block texture is 16x16

        n_blocks = len(self.block_atlas)

        self.block_atlas_tex = self._context.texture(
            (self.block_size[0] * 3, self.block_size[1] * n_blocks),
            3
        )
        self.block_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Block atlas size: {self.block_atlas_tex.width}x{self.block_atlas_tex.height}")

        for i, block_variant in enumerate(self.block_order):
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
                    self.block_size[0] * 0,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )
            self.block_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_size[0] * 1,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )
            self.block_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_size[0] * 2,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )
    
    def generate_emissive_atlas_texture(self) -> None:
        n_blocks = len(self.block_atlas)

        self.emissive_atlas_tex = self._context.texture(
            (self.block_size[0] * 3, self.block_size[1] * n_blocks),
            3
        )
        self.emissive_atlas_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        print(f"Emissive atlas size: {self.emissive_atlas_tex.width}x{self.emissive_atlas_tex.height}")

        for i, block_variant in enumerate(self.block_order):
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
                    self.block_size[0] * 0,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )
            self.emissive_atlas_tex.write(
                bottom_data,
                viewport=(
                    self.block_size[0] * 1,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )
            self.emissive_atlas_tex.write(
                side_data,
                viewport=(
                    self.block_size[0] * 2,
                    self.block_size[1] * (i-1),
                    self.block_size[0],
                    self.block_size[1]
                )
            )

    def render(self) -> None:
        """ Render one frame. """

        self._pathtracer_fbo.use()
        self._bluenoise_texture.use(0)
        self._voxel_tex.use(1)
        self._sky_texture.use(2)
        self.block_atlas_tex.use(3)
        self.emissive_atlas_tex.use(4)
        self._pt_vao.render()

        self._post_fbo.use()
        self._pathtracer_target_texture.use(0)
        self._post_vao.render()

        self._context.screen.use()
        self._post_target_texture.use(0)
        self._upscale_vao.render()