"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

import platform

import pygame
import moderngl
import imgui

from src import shared
from src.common import __version__, OGL_VERSION
from src.world import VoxelWorld
from src.renderer import Renderer
from src.gui import ImguiPygameModernGLAbomination
from src.scene import Scene
from src.platforminfo import get_cpu_info, get_gpu_info


class App:
    """
    Top-level application class.
    """

    def __init__(self,
            resolution: tuple[int, int],
            logical_resolution: tuple[int, int],
            target_fps: int = 300
            ) -> None:
        """
        Parameters
        ----------
        resolution
            Window resolution.
        logical_resolution
            Rendering resolution independent of window.
        target_fps
            Target framerate limit.
        """

        self._resolution = resolution
        self._logical_resolution = logical_resolution

        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, OGL_VERSION[0])
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, OGL_VERSION[1])

        pygame.display.set_mode(self._resolution, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(f"Voxel Pathtracer {__version__}")
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps
        self.dt = 0.25
        self.events: list[pygame.Event]

        shared.world = VoxelWorld((16, 16, 16))
        shared.renderer = Renderer(self._resolution, self._logical_resolution)
        shared.renderer.update_grid_texture()

        self.gui = ImguiPygameModernGLAbomination(self._resolution, shared.renderer._context)

        self.cpu_info = get_cpu_info()
        self.gpu_info = get_gpu_info(shared.renderer._context)

        self.python_version: str
        self.pygame_version: str
        self.sdl_version: str
        self.moderngl_version: str
        self.opengl_version: str
        self.imgui_version: str
        self.fetch_version_info()


        self.current_block = 1

        self.scenes = {}
        self.current_scene_name = ""

        self.is_running = False
        shared.app = self

    @property
    def logical_scale(self) -> float:
        # Assumes aspect ratio is the same.
        return self._logical_resolution[0] / self._resolution[0]
    
    @property
    def current_scene(self) -> Scene:
        return self.scenes[self.current_scene_name]
    
    def add_scene(self, scene: Scene, *args, **kwargs) -> None:
        scene_class_name = scene.__class__.__name__
        self.current_scene_name = scene_class_name
        self.scenes[scene_class_name] = scene(*args, **kwargs)
    
    def fetch_version_info(self) -> None:
        """ Gather dependency version information in MAJOR.MINOR.PATCH format. """

        self.python_version = platform.python_version()
        self.pygame_version = pygame.version.ver
        self.sdl_version = ".".join((str(v) for v in pygame.get_sdl_version()))
        self.moderngl_version = moderngl.__version__
        ogl_major = shared.renderer._context.info["GL_MAJOR_VERSION"]
        ogl_minor = shared.renderer._context.info["GL_MINOR_VERSION"]
        self.opengl_version = f"{ogl_major}.{ogl_minor}"
        self.imgui_version = imgui.__version__

    def run(self) -> None:
        self.is_running = True

        while self.is_running:
            self.dt = self.clock.tick(self.target_fps) * 0.001

            self.events = pygame.event.get()
            for event in self.events:
                if event.type == pygame.QUIT:
                    self.is_running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False
            
            # Only process UI when mouse is enabled
            # otherwise you can accidentally alter widgets when roaming around
            if not pygame.mouse.get_relative_mode():
                self.gui.process_events(self.events)

            self.current_scene.prev_camera = self.current_scene.camera.copy()
            self.current_scene.update()

            self.current_scene.render()

            imgui.render()
            self.gui.render(imgui.get_draw_data())

            pygame.display.flip()