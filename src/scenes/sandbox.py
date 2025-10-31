"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from math import pi
from time import perf_counter

import pygame
import imgui

from src import shared
from src.scene import Scene
from src.camera import CameraMode
from src.common import HIDE_HW_INFO, MAX_BOUNCES


class Sandbox(Scene):
    def __init__(self) -> None:
        super().__init__()

        pygame.mouse.set_relative_mode(True)

        # Cornell box camera
        self.camera.position = pygame.Vector3(137.0, 40.0 - 2.5 * 3, 37.5)
        self.camera.yaw = 180.0
        self.camera.pitch = 0.0
        #self.camera.fov = 23
        self.camera.update()

        self.mouse_sensitivity = 0.1
        self.movement_speed = 34.0

        self.color_profiles = ["Nice-ish", "Zeroes", "Noir"]
        self.current_color_profile = 0

        self.current_block = 1

        self.__acc = 0.0
        self.__acc_last = perf_counter()
        self.__rays_per_sec = 0.0

    def update(self) -> None:
        app = shared.app
        renderer = shared.renderer

        should_reset_acc = False

        for event in app.events:
            if event.type == pygame.MOUSEBUTTONDOWN:
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
                hitinfo = shared.world.dda(self.camera.position, direction)

                voxel = pygame.Vector3(*hitinfo.voxel)
                
                if mode == "place":
                    voxel += hitinfo.normal

                ivoxel = (int(voxel.x), int(voxel.y), int(voxel.z))

                if hitinfo.hit:
                    shared.world[ivoxel] = block
                    renderer.update_grid_texture()
                    should_reset_acc = True

            elif event.type == pygame.MOUSEBUTTONUP:
                # TODO: Resetting accumulation every mouse button event
                #       might not be desirable in the future, but for now
                #       it's good enough instead of checking every single
                #       UI update and world state change.
                if not pygame.mouse.get_relative_mode():
                    should_reset_acc = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LALT:
                    pygame.mouse.set_relative_mode(not pygame.mouse.get_relative_mode())

                elif event.key == pygame.K_F12:
                    renderer.high_quality_snapshot()

                elif event.key == pygame.K_F1:
                    shared.world.save("map.save")
                    should_reset_acc = True

                elif event.key == pygame.K_F2:
                    shared.world.load("map.save")
                    renderer.update_grid_texture()
                    should_reset_acc = True

                elif event.key == pygame.K_1:
                    self.current_block = 1
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_2:
                    self.current_block = 2
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_3:
                    self.current_block = 3
                    renderer.update_ui(self.current_block - 1)
                
                elif event.key == pygame.K_4:
                    self.current_block = 4
                    renderer.update_ui(self.current_block - 1)
                
                elif event.key == pygame.K_5:
                    self.current_block = 5
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_6:
                    self.current_block = 6
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_7:
                    self.current_block = 7
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_8:
                    self.current_block = 8
                    renderer.update_ui(self.current_block - 1)

                elif event.key == pygame.K_9:
                    self.current_block = 9
                    renderer.update_ui(self.current_block - 1)
        
        mouse_rel = pygame.Vector2(*pygame.mouse.get_rel())

        if pygame.mouse.get_relative_mode():
            rot = mouse_rel * self.mouse_sensitivity
            self.camera.yaw += rot.x

            if self.camera.mode == CameraMode.FIRST_PERSON:
                self.camera.pitch -= rot.y
            else:
                self.camera.pitch += rot.y

        else:
            # If the cursor mode is active and mouse is pressed,
            # it means the user is adjusting the UI
            # so reset accumulation to reflect the changes.
            if any(pygame.mouse.get_pressed()):
                should_reset_acc = True

        keys = pygame.key.get_pressed()

        if keys[pygame.K_r]:
            should_reset_acc = True

        mov = self.movement_speed * app.dt

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

        if should_reset_acc:
            renderer.clear_accumulation()

        if self.camera.mode == CameraMode.ORBIT:
            world_center = pygame.Vector3(
                shared.world.dimensions[0] * shared.world.voxel_size * 0.5,
                shared.world.dimensions[1] * shared.world.voxel_size * 0.5,
                shared.world.dimensions[2] * shared.world.voxel_size * 0.5
            )
            self.camera.target = world_center

        self.camera.update()
        self._update_camera_uniform()

    def render(self) -> None:
        app = shared.app
        renderer = shared.renderer

        renderer.render(ui=True)

        imgui.new_frame()
        imgui.begin("Sandbox Scene (press ALT to use UI)", True, flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        imgui.set_window_position(0, 0)

        if imgui.tree_node("Information", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            imgui.text(f"FPS: {round(app.clock.get_fps())}")
            imgui.text(f"Resolution: {app._resolution[0]}x{app._resolution[1]}")
            imgui.text(f"Renderer: {app._logical_resolution[0]}x{app._logical_resolution[1]} ({round(app.logical_scale, 2)}x)")

            if not HIDE_HW_INFO:
                imgui.text(f"CPU: {app.cpu_info['name']}")
                imgui.text(f"GPU: {app.gpu_info['name']}")

            imgui.text(f"Python:    {app.python_version}")
            imgui.text(f"Pygame-CE: {app.pygame_version}")
            imgui.text(f"SDL:       {app.sdl_version}")
            imgui.text(f"ModernGL:  {app.moderngl_version}")
            imgui.text(f"OpenGL:    {app.opengl_version}")
            imgui.text(f"ImGUI:     {app.imgui_version}")

            imgui.tree_pop()

        if imgui.tree_node("Statistics", imgui.TREE_NODE_FRAMED):
            _, renderer.settings.collect_information = imgui.checkbox("Collect stats (affects performance *heavily*)", renderer.settings.collect_information)

            if not renderer.settings.collect_information:
                imgui.text(f"Rays launched per frame: 0")
                imgui.text(f"Rays launched per second: 0")
                imgui.text(f"Primary rays launched: 0")
            else:
                imgui.text(f"Rays launched per frame: {round(renderer.frame_ray_count / 1000000.0, 1)} million")

                self.__acc += renderer.frame_ray_count
                
                now = perf_counter()
                if now - self.__acc_last > 1.0:
                    self.__acc_last = now
                    self.__rays_per_sec = self.__acc
                    self.__acc = 0.0

                rays_per_sec = self.__rays_per_sec / 1000000.0
                rays_per_sec_u = "million"
                if rays_per_sec > 1000.0:
                    rays_per_sec /= 1000.0
                    rays_per_sec_u = "billion"

                imgui.text(f"Rays launched per second: {round(rays_per_sec, 1)} {rays_per_sec_u}")

                primary_rays = renderer._logical_resolution[0] * renderer._logical_resolution[1] * renderer.settings.ray_count
                imgui.text(f"Primary rays launched: {round(primary_rays / 1000000.0, 1)} million")

            imgui.tree_pop()

        if imgui.tree_node("Color grading", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            _, renderer.settings.postprocessing = imgui.checkbox("Post-processing", renderer.settings.postprocessing)

            clicked, self.current_color_profile = imgui.combo(
                "Color profile", self.current_color_profile, self.color_profiles
            )

            if clicked:
                if self.current_color_profile == 0:
                    renderer.settings.color_profile_custom()
                elif self.current_color_profile == 1:
                    renderer.settings.color_profile_zero()
                elif self.current_color_profile == 2:
                    renderer.settings.color_profile_noir()

            tm_name = ("Clamp", "ACES Filmic")[renderer.settings.tonemapper]
            _, renderer.settings.tonemapper = imgui.slider_int(f"Tonemapper", renderer.settings.tonemapper, 0, 1, format=tm_name)

            _, renderer.settings.chromatic_aberration = imgui.slider_float("Chromatic Aberration", renderer.settings.chromatic_aberration, 0.000, 0.010, format="%.4f")
            _, renderer.settings.exposure  = imgui.slider_float("Exposure", renderer.settings.exposure, -5.0, 5.0, format="%.1f")
            _, renderer.settings.brightness = imgui.slider_float("Brightness", renderer.settings.brightness, -0.5, 0.5, format="%.4f")
            _, renderer.settings.contrast = imgui.slider_float("Contrast", renderer.settings.contrast, 0.0, 2.0, format="%.4f")
            _, renderer.settings.saturation = imgui.slider_float("Saturation", renderer.settings.saturation, 0.0, 2.5, format="%.4f")

            _, renderer.settings.enable_eye_adaptation = imgui.checkbox("Auto exposure adaptation", renderer.settings.enable_eye_adaptation)
            imgui.text(f"Adapted exposure: {round(renderer.settings.adapted_exposure, 3)}")
            imgui.text(f"Adaptation speed: {round(renderer.settings.adaptation_speed, 3)}")
            
            imgui.tree_pop()

        if imgui.tree_node("Path-tracing", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            spp_dec = imgui.arrow_button("decrease-samples", imgui.DIRECTION_LEFT)
            imgui.same_line()
            imgui.text(f"{renderer.settings.ray_count}")
            imgui.same_line()
            spp_inc = imgui.arrow_button("increase-samples", imgui.DIRECTION_RIGHT)
            imgui.same_line()
            imgui.text("Rays/pixel")

            if spp_inc:
                renderer.settings.increase_ray_counts()

            if spp_dec:
                renderer.settings.decrease_ray_counts()

            _, renderer.settings.bounces = imgui.slider_int(f"Bounces", renderer.settings.bounces, 1, MAX_BOUNCES)

            noise_name = ("Mulberry32 PRNG", "Heitz Bluenoise")[renderer.settings.noise_method-1]
            _, renderer.settings.noise_method = imgui.slider_int(f"Noise method", renderer.settings.noise_method, 1, 2, format=noise_name)

            _, renderer.settings.russian_roulette = imgui.checkbox("Russian-roulette", renderer.settings.russian_roulette)
            _, renderer.settings.enable_accumulation = imgui.checkbox("Temporal accumulation", renderer.settings.enable_accumulation)
            _, renderer.settings.nee = imgui.checkbox("Next Event Estimation", renderer.settings.nee)

            aa_name = ("None", "Jitter Sampling", "FXAA")[renderer.settings.antialiasing]
            _, renderer.settings.antialiasing = imgui.slider_int(f"Anti-aliasing", renderer.settings.antialiasing, 0, 2, format=aa_name)

            up_name = ("Nearest", "Bilinear", "Bicubic")[renderer.settings.upscaling_method]
            _, renderer.settings.upscaling_method = imgui.slider_int(f"Upscaler", renderer.settings.upscaling_method, 0, 2, format=up_name)

            out_name = ("GI", "Normals", "Bounces")[renderer.settings.pathtracer_output]
            _, renderer.settings.pathtracer_output = imgui.slider_int(f"Out texture", renderer.settings.pathtracer_output, 0, 2, format=out_name)

            if imgui.tree_node("High-quality render settings"):
                _, renderer.settings.highquality_ray_count = imgui.slider_int(f"Rays/pixel", renderer.settings.highquality_ray_count, 512, 2048)
                _, renderer.settings.highquality_bounces = imgui.slider_int(f"Bounces", renderer.settings.highquality_bounces, 5, 35)

                imgui.tree_pop()

            imgui.tree_pop()

        if imgui.tree_node("Denoising", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            clicked, selected_denoiser = imgui.combo(
                "Denoiser", renderer.settings.denoiser_id, renderer.settings.denoisers
            )
            if clicked:
                renderer.settings.denoiser_id = selected_denoiser

            if selected_denoiser == 1:
                _, renderer._denoise_program["u_hw"] = imgui.slider_int(f"u_hw", renderer._denoise_program["u_hw"].value, 1, 14)
                _, renderer._denoise_program["u_sigmaspace"] = imgui.slider_float(f"u_sigmaspace", renderer._denoise_program["u_sigmaspace"].value, 0.1, 50.0)
                _, renderer._denoise_program["u_sigmacolor"] = imgui.slider_float(f"u_sigmacolor", renderer._denoise_program["u_sigmacolor"].value, 0.1, 50.0)

            imgui.tree_pop()

        if imgui.tree_node("Sky", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            #_, renderer.settings.sun_radiance = imgui.slider_float(f"Sun radiance", renderer.settings.sun_radiance, 0.0, 5000.0)
            _, renderer.settings.sky_turbidity = imgui.slider_float(f"Sky turbidity", renderer.settings.sky_turbidity, 2.0, 10.0)
            _, renderer.settings.sun_angular_radius = imgui.slider_float(f"Sun angular radius", renderer.settings.sun_angular_radius, 0.0, pi * 0.05)
            _, renderer.settings.sun_azimuth = imgui.slider_float(f"Sun azimuth", renderer.settings.sun_azimuth, 0.0, 360.0)
            _, renderer.settings.sun_altitude = imgui.slider_float(f"Sun altitude", renderer.settings.sun_altitude, 0.0, 90.0)

            _, renderer.settings.enable_sky_texture = imgui.checkbox("Use sky texture", renderer.settings.enable_sky_texture)

            imgui.tree_pop()

        if imgui.tree_node("Camera", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            _, self.camera.fov = imgui.slider_float("FOV", self.camera.fov, 0.0, 180.0, format="%.4f")
            _, self.mouse_sensitivity = imgui.slider_float("Sensitivity", self.mouse_sensitivity, 0.01, 0.3, format="%.4f")
            _, camera_mode_int = imgui.slider_int("Mode", self.camera.mode.value, 0, 1, format=self.camera.mode.name.lower().replace("_", " ").capitalize())
            self.camera.mode = CameraMode(camera_mode_int)

            imgui.tree_pop()

        if imgui.tree_node("Controls", imgui.TREE_NODE_FRAMED):
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