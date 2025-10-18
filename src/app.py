"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from math import floor
from dataclasses import dataclass

import pygame
import imgui

from src.camera import Camera
from src.renderer import Renderer
from src.gui import ImguiPygameModernGLAbomination


def vec3_sign(v: pygame.Vector3) -> pygame.Vector3:
    return pygame.Vector3(
        int(v.x > 0) - int(v.x < 0),
        int(v.y > 0) - int(v.y < 0),
        int(v.z > 0) - int(v.z < 0)
    )


@dataclass
class HitInfo:
    hit: bool
    point: pygame.Vector3
    normal: pygame.Vector3
    voxel: tuple[int, int, int]


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
        pygame.display.set_caption("Voxel Pathtracer Prototype  -  Pygame-CE & ModernGL")
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps

        self.camera = Camera(logical_resolution)

        self.renderer = Renderer(self.resolution, self.logical_resolution)

        self.gui = ImguiPygameModernGLAbomination(self.resolution, self.renderer._context)

        self.is_running = False

        # floor
        for z in range(self.renderer.grid_size[2]):
            for x in range(self.renderer.grid_size[0]):
                self.renderer.grid[(x, 0, z)] = 4

        self.renderer.update_grid()

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

    def dda(self, origin: pygame.Vector3, dir: pygame.Vector3) -> HitInfo:
        # defined in shader
        VOXEL_SIZE = 5.0
        MAX_DDA_STEPS = 34
        
        voxel = origin / VOXEL_SIZE
        voxel.x = floor(voxel.x)
        voxel.y = floor(voxel.y)
        voxel.z = floor(voxel.z)

        step = vec3_sign(dir)

        next_boundary = pygame.Vector3(
            (voxel.x + (1.0 if step.x > 0.0 else 0.0)) * VOXEL_SIZE,
            (voxel.y + (1.0 if step.y > 0.0 else 0.0)) * VOXEL_SIZE,
            (voxel.z + (1.0 if step.z > 0.0 else 0.0)) * VOXEL_SIZE
        )

        t_max = next_boundary - origin
        if (dir.x == 0.0): t_max.x = float("inf")
        else: t_max.x /= dir.x
        if (dir.y == 0.0): t_max.y = float("inf")
        else: t_max.y /= dir.y
        if (dir.z == 0.0): t_max.z = float("inf")
        else: t_max.z /= dir.z

        t_delta = pygame.Vector3(0.0)
        if dir.x == 0.0: t_delta.x = float("inf")
        else: t_delta.x = abs(VOXEL_SIZE / dir.x)
        if dir.y == 0.0: t_delta.y = float("inf")
        else: t_delta.y = abs(VOXEL_SIZE / dir.y)
        if dir.z == 0.0: t_delta.z = float("inf")
        else: t_delta.z = abs(VOXEL_SIZE / dir.z)

        normal = pygame.Vector3(0.0)

        # traverse
        for i in range(MAX_DDA_STEPS):
            hit_t = min(t_max.x, t_max.y, t_max.z)
            
            if t_max.x < t_max.y and t_max.x < t_max.z:
                voxel.x += step.x
                t_max.x += t_delta.x
                normal = pygame.Vector3(-step.x, 0.0, 0.0)
            
            elif t_max.y < t_max.z:
                voxel.y += step.y
                t_max.y += t_delta.y
                normal = pygame.Vector3(0.0, -step.y, 0.0)

            else:
                voxel.z += step.z
                t_max.z += t_delta.z
                normal = pygame.Vector3(0.0, 0.0, -step.z)

            ivoxel = (int(voxel.x), int(voxel.y), int(voxel.z))
            sample = self.renderer.grid[ivoxel]
            if sample > 0:
                return HitInfo(
                    True,
                    origin + dir * hit_t,
                    normal,
                    ivoxel
                )
            
        return HitInfo(False, pygame.Vector3(), pygame.Vector3(), (0, 0, 0))


    def run(self) -> None:
        self.is_running = True

        pygame.mouse.set_relative_mode(True)

        voxel_x = 0
        voxel_y = 0
        voxel_z = 0

        while self.is_running:
            dt = self.clock.tick(self.target_fps) * 0.001

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.is_running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pass

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False

                    elif event.key == pygame.K_LALT:
                        pygame.mouse.set_relative_mode(False)

                    elif event.key == pygame.K_LEFT:
                        voxel_x -= 1
                    
                    elif event.key == pygame.K_RIGHT:
                        voxel_x += 1

                    elif event.key == pygame.K_UP:
                        voxel_y -= 1
                    
                    elif event.key == pygame.K_DOWN:
                        voxel_y += 1

                    elif event.key == pygame.K_o:
                        voxel_z -= 1
                    
                    elif event.key == pygame.K_l:
                        voxel_z += 1

                    elif event.key == pygame.K_RETURN:
                        self.renderer.grid[(voxel_x, voxel_y, voxel_z)] = 255
                        self.renderer.update_grid()
                
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LALT:
                        pygame.mouse.set_relative_mode(True)

            self.gui.process_events(events)

            rx, ry = pygame.mouse.get_rel()

            if pygame.mouse.get_relative_mode():
                sx = -rx * 0.1
                sy = ry * 0.1
                self.camera.rotate_yaw(sx)
                self.camera.rotate_pitch(sy)

            keys = pygame.key.get_pressed()

            mov = 40.0 * dt

            if keys[pygame.K_w]:
                self.camera.move(mov)

            if keys[pygame.K_s]:
                self.camera.move(-mov)

            if keys[pygame.K_a]:
                self.camera.strafe(-mov)

            if keys[pygame.K_d]:
                self.camera.strafe(mov)

            if keys[pygame.K_e]:
                self.camera.position -= pygame.Vector3(0.0, mov, 0.0)
                self.camera.look_at -= pygame.Vector3(0.0, mov, 0.0)

            if keys[pygame.K_q]:
                self.camera.position += pygame.Vector3(0.0, mov, 0.0)
                self.camera.look_at += pygame.Vector3(0.0, mov, 0.0)

            self.camera.update()
            self._update_camera_uniform()

            #self.renderer.set_voxel((voxel_x, voxel_y, voxel_z))
            self.renderer.render()

            imgui.new_frame()
            imgui.begin("Nice window", True, flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.set_window_position(0, 0)

            imgui.text(f"FPS: {round(self.clock.get_fps())}")
            imgui.text(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
            imgui.text(f"Renderer: {self.logical_resolution[0]}x{self.logical_resolution[1]} ({round(self.logical_scale, 2)}x)")
            imgui.text(f"Voxel: {voxel_x}, {voxel_y}, {voxel_z}")

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