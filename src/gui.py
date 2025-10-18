"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from typing import Optional

import ctypes

import pygame
import moderngl
import imgui


class ImguiPygameModernGLAbomination:
    """
    Processes pygame events, handles ModernGL rendering and ImGUI IO.
    """

    def __init__(self,
            display_size: tuple[int, int],
            context: moderngl.Context,
            vertex_shader: Optional[str] = None,
            fragment_shader: Optional[str] = None
            ) -> None:
        """
        Parameters
        ----------
        display_size
            ImGUI display size.
        context
            ModernGL context created.
        vertex_shader
            Custom vertex shader. Uses the default one if not provided.
        fragment_shader
            Custom fragment shader. Uses the default one if not provided.
        """

        self._context = context

        imgui.create_context()
        self.io = imgui.get_io()

        self.io.display_size = display_size
        self.io.delta_time = 1.0 / 60.0

        self._textures = {}
        self.refresh_font_texture()

        base_vertex_shader = """
        #version 330
        in vec2 in_position;
        in vec2 in_uv;
        in vec4 in_color;
        out vec2 v_uv;
        out vec4 v_color;
        uniform mat4 u_proj;
        void main() {
            gl_Position = u_proj * vec4(in_position, 0.0, 1.0);
            v_uv = in_uv;
            v_color = in_color;
        }
        """

        base_fragment_shader = """
        #version 330
        in vec2 v_uv;
        in vec4 v_color;
        out vec4 f_color;
        uniform sampler2D s_texture;
        void main() {
            f_color = v_color * texture(s_texture, v_uv);
        }
        """

        self._program = self._context.program(
            vertex_shader=vertex_shader if vertex_shader else base_vertex_shader,
            fragment_shader=fragment_shader if fragment_shader else base_fragment_shader,
        )

        # Taken from moderngl_window's integration, not sure why we multiply by 65536
        self._vbo = self._context.buffer(reserve=imgui.VERTEX_SIZE * 65536)
        self._ibo = self._context.buffer(reserve=imgui.INDEX_SIZE * 65536)
        
        self._proj = self._program["u_proj"]

        self._vao = self._context.vertex_array(
            self._program,
            [
                (self._vbo, "2f 2f 4f1", "in_position", "in_uv", "in_color")
            ],
            index_buffer=self._ibo,
            index_element_size=imgui.INDEX_SIZE,
        )

    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """ Cleanup resources used. """

        for texture in self._textures:
            self._textures[texture].release()

        self._program.release()
        self._vbo.release()
        self._ibo.release()
        self._vao.release()

    def load_font(self, filepath: str, size: float) -> imgui.core._Font:
        """
        Load custom TTF/OTF font into ImGUI.
        Returns the loaded font object to be used when rendering.
        
        Parameters
        ----------
        filepath
            Path to the font file.
        size
            Glyph size in pixels.
        """
        
        new_font = self.io.fonts.add_font_from_file_ttf(filepath, size)
        self.refresh_font_texture()
        return new_font
    
    def refresh_font_texture(self) -> None:
        """ Refresh the ImGUI font as a texture. """

        width, height, pixels = self.io.fonts.get_tex_data_as_rgba32()

        font_texture = self._context.texture((width, height), 4, pixels)
        self._textures[font_texture.glo] = font_texture

        self.io.fonts.texture_id = font_texture.glo
        self.io.fonts.clear_tex_data()

    def resize(self, display_size: tuple[int, int]) -> None:
        """
        Resize ImGUI display size.
        Parameters
        ----------
        display_size
            Tuple of new resolution.
        """
        self.io.display_size = display_size

    def process_events(self, events: list[pygame.Event]) -> None:
        """
        Process pygame events and update ImGUI IO.
        
        Parameters
        ----------
        events
            List of pygame events (most likely from `pygame.event.get`) 
        """

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.io.mouse_down[event.button - 1] = True

            if event.type == pygame.MOUSEBUTTONUP:
                self.io.mouse_down[event.button - 1] = False

        self.io.mouse_pos = pygame.mouse.get_pos()

    def render(self, draw_data) -> None:
        """
        Render ImGUI onto current framebuffer.
        Note: This changes blending modes.
        Parameters
        ----------
        draw_data
            ImGUI draw commands (`imgui.get_draw_data`)
        """

        display_width = self.io.display_size.x
        display_height = self.io.display_size.y

        # Thanks to moderngl_window, imgui's draw commands are very low level

        self._context.enable_only(moderngl.BLEND)
        self._context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._context.blend_equation = moderngl.FUNC_ADD

        draw_data.scale_clip_rects(1.0, 1.0)

        self._proj.value = (
            2.0 / display_width, 0.0,                   0.0,  0.0,
            0.0,                 2.0 / -display_height, 0.0,  0.0,
            0.0,                 0.0,                   -1.0, 0.0,
            -1.0,                1.0,                   0.0,  1.0,
        )

        for commands in draw_data.commands_lists:
            # Write the vertex and index buffer data without copying it
            vtx_type = ctypes.c_byte * commands.vtx_buffer_size * imgui.VERTEX_SIZE
            idx_type = ctypes.c_byte * commands.idx_buffer_size * imgui.INDEX_SIZE
            vtx_arr = (vtx_type).from_address(commands.vtx_buffer_data)
            idx_arr = (idx_type).from_address(commands.idx_buffer_data)
            self._vbo.write(vtx_arr)
            self._ibo.write(idx_arr)

            idx_pos = 0
            for command in commands.commands:
                # Texture's command id = moderngl `glo` attribute
                texture = self._textures[command.texture_id]
                texture.use(0)
            
                x, y, z, w = command.clip_rect
                self._context.scissor = int(x), int(display_height - w), int(z - x), int(w - y)
                self._vao.render(moderngl.TRIANGLES, vertices=command.elem_count, first=idx_pos)
                idx_pos += command.elem_count

        self._context.scissor = None