"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

import pygame
import imgui

from src.camera import Camera
from src.renderer import Renderer
from src.gui import ImguiPygameModernGLAbomination


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
        self.resolution = resolution
        self.logical_resolution = logical_resolution

        pygame.display.set_mode(self.resolution, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Voxel Pathtracer Prototype")
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps

        self.camera = Camera(logical_resolution)

        self.renderer = Renderer(self.resolution, self.logical_resolution)

        self.gui = ImguiPygameModernGLAbomination(self.resolution, self.renderer._context)

        self.is_running = False

    @property
    def logical_scale(self) -> float:
        # Assumes aspect ratio is the same.
        return self.logical_resolution[0] / self.resolution[0]

    def _update_camera_uniform(self) -> None:
        self.renderer._pt_program["u_camera.position"] = (
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        )
        self.renderer._pt_program["u_camera.center"] = (
            self.camera.center.x,
            self.camera.center.y,
            self.camera.center.z
        )
        self.renderer._pt_program["u_camera.u"] = (
            self.camera.u.x,
            self.camera.u.y,
            self.camera.u.z
        )
        self.renderer._pt_program["u_camera.v"] = (
            self.camera.v.x,
            self.camera.v.y,
            self.camera.v.z
        )

    def run(self) -> None:
        self.is_running = True

        while self.is_running:
            dt = self.clock.tick(self.target_fps) * 0.001

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.is_running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False

            self.gui.process_events(events)

            rx, ry = pygame.mouse.get_rel()

            if pygame.mouse.get_pressed()[2]:
                pygame.mouse.set_relative_mode(True)
                sx = -rx * 0.1
                sy = ry * 0.1
                self.camera.rotate_yaw(sx)
                self.camera.rotate_pitch(sy)
            else:
                pygame.mouse.set_relative_mode(False)

            keys = pygame.key.get_pressed()

            mov = 20.0 * dt

            if keys[pygame.K_w]:
                self.camera.move(mov)

            if keys[pygame.K_s]:
                self.camera.move(-mov)

            if keys[pygame.K_a]:
                self.camera.strafe(-mov)

            if keys[pygame.K_d]:
                self.camera.strafe(mov)

            if keys[pygame.K_q]:
                self.camera.position -= pygame.Vector3(0.0, mov, 0.0)
                self.camera.look_at -= pygame.Vector3(0.0, mov, 0.0)

            if keys[pygame.K_e]:
                self.camera.position += pygame.Vector3(0.0, mov, 0.0)
                self.camera.look_at += pygame.Vector3(0.0, mov, 0.0)

            self.camera.update()
            self._update_camera_uniform()

            self.renderer.render()

            imgui.new_frame()
            imgui.begin("Nice window", True, flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.set_window_position(0, 0)

            imgui.text(f"FPS: {round(self.clock.get_fps())}")
            imgui.text(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
            imgui.text(f"Renderer: {self.logical_resolution[0]}x{self.logical_resolution[1]} ({round(self.logical_scale, 2)}x)")

            if imgui.tree_node("Post-processing", imgui.TREE_NODE_DEFAULT_OPEN):
                _, self.renderer.settings.postprocessing = imgui.checkbox("Enable post-processing", self.renderer.settings.postprocessing)
                _, self.renderer.settings.exposure  = imgui.slider_float("Exposure", self.renderer.settings.exposure, -5.0, 5.0, format="%.1f")

                tm_name = ("None", "ACES Filmic")[self.renderer.settings.tonemapper]
                _, self.renderer.settings.tonemapper = imgui.slider_int(f"Tonemapper", self.renderer.settings.tonemapper, 0, 1, format=tm_name)

                _, self.renderer.settings.brightness = imgui.slider_float("Brightness", self.renderer.settings.brightness, -0.5, 0.5, format="%.4f")
                _, self.renderer.settings.contrast = imgui.slider_float("Contrast", self.renderer.settings.contrast, 0.0, 1.2, format="%.4f")
                _, self.renderer.settings.saturation = imgui.slider_float("Saturation", self.renderer.settings.saturation, 0.0, 1.75, format="%.4f")
                
                imgui.tree_pop()

            if imgui.tree_node("Path-tracing", imgui.TREE_NODE_DEFAULT_OPEN):
                _, self.renderer.settings.ray_count = imgui.slider_int(f"Rays/pixel", self.renderer.settings.ray_count, 1, 30)
                _, self.renderer.settings.bounces = imgui.slider_int(f"Bounces", self.renderer.settings.bounces, 1, 5)
                noise_name = ("None", "Mulberry32", "Bluenoise")[self.renderer.settings.noise_method]
                _, self.renderer.settings.noise_method = imgui.slider_int(f"Noise method", self.renderer.settings.noise_method, 0, 2, format=noise_name)
                imgui.tree_pop()

            imgui.end()
            
            imgui.render()

            self.gui.render(imgui.get_draw_data())

            pygame.display.flip()