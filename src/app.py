"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

import platform

import pygame
import moderngl
import imgui

from src.common import __version__, MAX_BOUNCES, HIDE_HW_INFO
from src.camera import Camera, CameraMode
from src.world import VoxelWorld
from src.renderer import Renderer
from src.gui import ImguiPygameModernGLAbomination
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

        pygame.display.set_mode(self._resolution, pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(f"Voxel Pathtracer {__version__}")
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps

        self.camera = Camera(self._logical_resolution)
        self.world = VoxelWorld((16, 16, 16))
        self.renderer = Renderer(self._resolution, self._logical_resolution, self.world)
        self.gui = ImguiPygameModernGLAbomination(self._resolution, self.renderer._context)

        self.camera.position = pygame.Vector3(40, 30, -20)
        self.camera.yaw = 90
        self.camera.pitch = -30
        self.camera.update()
        self.prev_camera: Camera

        self.cpu_info = get_cpu_info()
        self.gpu_info = get_gpu_info(self.renderer._context)

        self.python_version: str
        self.pygame_version: str
        self.sdl_version: str
        self.moderngl_version: str
        self.opengl_version: str
        self.imgui_version: str
        self.fetch_version_info()

        self.renderer.update_grid_texture()

        self.current_block = 1

        self.is_running = False

    @property
    def logical_scale(self) -> float:
        # Assumes aspect ratio is the same.
        return self._logical_resolution[0] / self._resolution[0]
    
    def fetch_version_info(self) -> None:
        """ Gather dependency version information in MAJOR.MINOR.PATCH format. """

        self.python_version = platform.python_version()
        self.pygame_version = pygame.version.ver
        self.sdl_version = ".".join((str(v) for v in pygame.get_sdl_version()))
        self.moderngl_version = moderngl.__version__
        ogl_major = self.renderer._context.info["GL_MAJOR_VERSION"]
        ogl_minor = self.renderer._context.info["GL_MINOR_VERSION"]
        self.opengl_version = f"{ogl_major}.{ogl_minor}"
        self.imgui_version = imgui.__version__

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

        self.renderer._pt_program["u_prev_camera.position"] = (
            self.prev_camera.position.x,
            self.prev_camera.position.y,
            self.prev_camera.position.z
        )
        self.renderer._pt_program["u_prev_camera.center"] = (
            self.prev_camera.center.x,
            self.prev_camera.center.y,
            self.prev_camera.center.z
        )
        self.renderer._pt_program["u_prev_camera.u"] = (
            self.prev_camera.u.x,
            self.prev_camera.u.y,
            self.prev_camera.u.z
        )
        self.renderer._pt_program["u_prev_camera.v"] = (
            self.prev_camera.v.x,
            self.prev_camera.v.y,
            self.prev_camera.v.z
        )

    def run(self) -> None:
        self.is_running = True

        pygame.mouse.set_relative_mode(True)
        mouse_sensitivity = 0.1

        color_profiles = ["Nice-ish", "Zeroes"]
        current_color_profile = 0

        while self.is_running:
            dt = self.clock.tick(self.target_fps) * 0.001

            should_reset_acc = False

            self.prev_camera = self.camera.copy()

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.is_running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not pygame.mouse.get_relative_mode(): continue

                    if event.button == 1:
                        mode = "break"
                    elif event.button == 3:
                        mode = "place"
                    else:
                        continue

                    if mode == "break":
                        block = 0
                    else:
                        block = self.current_block

                    direction = self.camera.front
                    hitinfo = self.world.dda(self.camera.position, direction)

                    voxel = pygame.Vector3(*hitinfo.voxel)
                    
                    if mode == "place":
                        voxel += hitinfo.normal

                    ivoxel = (int(voxel.x), int(voxel.y), int(voxel.z))

                    if hitinfo.hit:
                        self.world[ivoxel] = block
                        self.renderer.update_grid_texture()
                        should_reset_acc = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    # TODO: Resetting accumulation every mouse button event
                    #       might not be desirable in the future, but for now
                    #       it's good enough instead of checking every single
                    #       UI update and world state change.
                    if not pygame.mouse.get_relative_mode():
                        should_reset_acc = True

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_running = False

                    elif event.key == pygame.K_LALT:
                        pygame.mouse.set_relative_mode(not pygame.mouse.get_relative_mode())

                    elif event.key == pygame.K_F12:
                        self.renderer.high_quality_snapshot()

                    elif event.key == pygame.K_F1:
                        self.world.save("map.save")
                        should_reset_acc = True

                    elif event.key == pygame.K_F2:
                        self.world.load("map.save")
                        self.renderer.update_grid_texture()
                        should_reset_acc = True

                    elif event.key == pygame.K_1:
                        self.current_block = 1
                        self.renderer.update_ui(self.current_block - 1)

                    elif event.key == pygame.K_2:
                        self.current_block = 2
                        self.renderer.update_ui(self.current_block - 1)

                    elif event.key == pygame.K_3:
                        self.current_block = 3
                        self.renderer.update_ui(self.current_block - 1)
                    
                    elif event.key == pygame.K_4:
                        self.current_block = 4
                        self.renderer.update_ui(self.current_block - 1)
                    
                    elif event.key == pygame.K_5:
                        self.current_block = 5
                        self.renderer.update_ui(self.current_block - 1)

                    elif event.key == pygame.K_6:
                        self.current_block = 6
                        self.renderer.update_ui(self.current_block - 1)

                    elif event.key == pygame.K_7:
                        self.current_block = 7
                        self.renderer.update_ui(self.current_block - 1)

            # Only process UI when mouse is enabled
            # otherwise you can accidentally alter widgets when roaming around
            if not pygame.mouse.get_relative_mode():
                self.gui.process_events(events)

            mouse_rel = pygame.Vector2(*pygame.mouse.get_rel())

            if pygame.mouse.get_relative_mode():
                rot = mouse_rel * mouse_sensitivity
                self.camera.yaw += rot.x

                if self.camera.mode == CameraMode.FIRST_PERSON:
                    self.camera.pitch -= rot.y
                else:
                    self.camera.pitch += rot.y

                if abs(mouse_rel.x) > 0.0 or abs(mouse_rel.y) > 0.0:
                    #should_reset_acc = True
                    pass

            else:
                # If the cursor mode is active and mouse is pressed,
                # it means the user is adjusting the UI
                # so reset accumulation to reflect the changes.
                if any(pygame.mouse.get_pressed()):
                    should_reset_acc = True

            keys = pygame.key.get_pressed()

            if keys[pygame.K_r]:
                should_reset_acc = True

            mov = 40.0 * dt

            if keys[pygame.K_w]:
                if self.camera.mode == CameraMode.FIRST_PERSON:
                    self.camera.move(mov)
                else:
                    self.camera.distance -= mov * 2.0

            if keys[pygame.K_s]:
                if self.camera.mode == CameraMode.FIRST_PERSON:
                    self.camera.move(-mov)
                else:
                    self.camera.distance += mov * 2.0

            if keys[pygame.K_a]:
                self.camera.strafe(-mov)

            if keys[pygame.K_d]:
                self.camera.strafe(mov)

            if keys[pygame.K_e]:
                self.camera.position -= self.camera.up * mov

            if keys[pygame.K_q]:
                self.camera.position += self.camera.up * mov

            if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_e] or keys[pygame.K_q]:...
                #should_reset_acc = True

            if should_reset_acc:
                self.renderer.settings.acc_frame = 0

            if self.camera.mode == CameraMode.ORBIT:
                world_center = pygame.Vector3(
                    self.world.dimensions[0] * self.world.voxel_size * 0.5,
                    self.world.dimensions[1] * self.world.voxel_size * 0.5,
                    self.world.dimensions[2] * self.world.voxel_size * 0.5
                )
                self.camera.target = world_center

            self.camera.update()
            self._update_camera_uniform()

            self.renderer.render()

            imgui.new_frame()
            imgui.begin("Settings (press ALT)", True, flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.set_window_position(0, 0)

            if imgui.tree_node("Information", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                imgui.text(f"FPS: {round(self.clock.get_fps())}")
                imgui.text(f"Resolution: {self._resolution[0]}x{self._resolution[1]}")
                imgui.text(f"Renderer: {self._logical_resolution[0]}x{self._logical_resolution[1]} ({round(self.logical_scale, 2)}x)")
                imgui.text(f"Accumulation: {self.renderer.settings.acc_frame}")

                if not HIDE_HW_INFO:
                    imgui.text(f"CPU: {self.cpu_info['name']}")
                    imgui.text(f"GPU: {self.gpu_info['name']}")

                imgui.text(f"Python:    {self.python_version}")
                imgui.text(f"Pygame-CE: {self.pygame_version}")
                imgui.text(f"SDL:       {self.sdl_version}")
                imgui.text(f"ModernGL:  {self.moderngl_version}")
                imgui.text(f"OpenGL:    {self.opengl_version}")
                imgui.text(f"ImGUI:     {self.imgui_version}")

                imgui.tree_pop()

            if imgui.tree_node("Color grading", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                _, self.renderer.settings.postprocessing = imgui.checkbox("Enable post-processing", self.renderer.settings.postprocessing)

                clicked, current_color_profile = imgui.combo(
                    "Color profile", current_color_profile, color_profiles
                )

                if clicked:
                    if current_color_profile == 0:
                        self.renderer.settings.color_profile_custom()
                    elif current_color_profile == 1:
                        self.renderer.settings.color_profile_zero()

                tm_name = ("None", "ACES Filmic")[self.renderer.settings.tonemapper]
                _, self.renderer.settings.tonemapper = imgui.slider_int(f"Tonemapper", self.renderer.settings.tonemapper, 0, 1, format=tm_name)

                _, self.renderer.settings.exposure  = imgui.slider_float("Exposure", self.renderer.settings.exposure, -5.0, 5.0, format="%.1f")
                _, self.renderer.settings.brightness = imgui.slider_float("Brightness", self.renderer.settings.brightness, -0.5, 0.5, format="%.4f")
                _, self.renderer.settings.contrast = imgui.slider_float("Contrast", self.renderer.settings.contrast, 0.0, 1.2, format="%.4f")
                _, self.renderer.settings.saturation = imgui.slider_float("Saturation", self.renderer.settings.saturation, 0.0, 1.75, format="%.4f")

                _, self.renderer.settings.enable_eye_adaptation = imgui.checkbox("Enable eye adaptation", self.renderer.settings.enable_eye_adaptation)
                imgui.text(f"Adapted exposure: {round(self.renderer.settings.adapted_exposure, 3)}")
                imgui.text(f"Adaptation speed: {round(self.renderer.settings.adaptation_speed, 3)}")
                
                imgui.tree_pop()

            if imgui.tree_node("Path-tracing", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                #_, self.renderer.settings.ray_count = imgui.slider_int(f"Rays/pixel", self.renderer.settings.ray_count, 1, MAX_RAYS_PER_PIXEL)

                spp_dec = imgui.arrow_button("decrease-samples", imgui.DIRECTION_LEFT)
                imgui.same_line()
                imgui.text(f"{self.renderer.settings.ray_count}")
                imgui.same_line()
                spp_inc = imgui.arrow_button("increase-samples", imgui.DIRECTION_RIGHT)
                imgui.same_line()
                imgui.text("Rays/pixel")

                if spp_inc:
                    self.renderer.settings.increase_ray_counts()

                if spp_dec:
                    self.renderer.settings.decrease_ray_counts()

                _, self.renderer.settings.bounces = imgui.slider_int(f"Bounces", self.renderer.settings.bounces, 1, MAX_BOUNCES)

                noise_name = ("None", "Mulberry32 PRNG", "Heitz Bluenoise")[self.renderer.settings.noise_method]
                _, self.renderer.settings.noise_method = imgui.slider_int(f"Noise method", self.renderer.settings.noise_method, 0, 2, format=noise_name)

                _, self.renderer.settings.russian_roulette = imgui.checkbox("Enable russian-roulette", self.renderer.settings.russian_roulette)
                _, self.renderer.settings.enable_accumulation = imgui.checkbox("Enable accumulation", self.renderer.settings.enable_accumulation)

                aa_name = ("None", "Jitter Sampling", "FXAA")[self.renderer.settings.antialiasing]
                _, self.renderer.settings.antialiasing = imgui.slider_int(f"Anti-aliasing", self.renderer.settings.antialiasing, 0, 2, format=aa_name)

                up_name = ("Nearest", "Bilinear", "Bicubic")[self.renderer.settings.upscaling_method]
                _, self.renderer.settings.upscaling_method = imgui.slider_int(f"Upscaler", self.renderer.settings.upscaling_method, 0, 2, format=up_name)

                if imgui.tree_node("High-quality render settings"):
                    _, self.renderer.settings.highquality_ray_count = imgui.slider_int(f"Rays/pixel", self.renderer.settings.highquality_ray_count, 512, 2048)
                    _, self.renderer.settings.highquality_bounces = imgui.slider_int(f"Bounces", self.renderer.settings.highquality_bounces, 5, 30)

                    imgui.tree_pop()

                imgui.tree_pop()

            if imgui.tree_node("Sky", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                _, self.renderer.settings.enable_sky_texture = imgui.checkbox("Enable sky texture", self.renderer.settings.enable_sky_texture)
                _, self.renderer.settings.sky_color = imgui.color_edit3("Sky color", *self.renderer.settings.sky_color, imgui.COLOR_EDIT_NO_INPUTS)

                imgui.tree_pop()

            if imgui.tree_node("Camera", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                _, self.camera.fov = imgui.slider_float("FOV", self.camera.fov, 0.0, 180.0, format="%.4f")
                _, mouse_sensitivity = imgui.slider_float("Sensitivity", mouse_sensitivity, 0.01, 0.3, format="%.4f")
                _, camera_mode_int = imgui.slider_int("Mode", self.camera.mode.value, 0, 1, format=self.camera.mode.name.lower().replace("_", " ").capitalize())
                self.camera.mode = CameraMode(camera_mode_int)

                imgui.tree_pop()

            if imgui.tree_node("Controls", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
                imgui.text(f"[LMB] to break voxels")
                imgui.text(f"[RMB] to place voxels")
                imgui.text(f"[WASD] to move around")
                imgui.text(f"[Q & E] to move vertically")
                imgui.text(f"[1 - 9] to change current block")
                imgui.text(f"[R] to reset accumulation")
                imgui.text(f"[ALT] to use toggle cursor and use UI")
                imgui.text(f"[F12] to take high-quality render snapshot")
                imgui.text(f"[F1] to save map onto 'map.save' file")
                imgui.text(f"[F2] to load map from 'map.save' file")
                imgui.text(f"[ESC] to quit")

                imgui.tree_pop()

            imgui.end()
            
            imgui.render()

            self.gui.render(imgui.get_draw_data())

            pygame.display.flip()