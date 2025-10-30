"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from math import pi

import pygame
import imgui

from src import shared
from src.scene import Scene
from src.camera import CameraMode
from src.common import HIDE_HW_INFO, MAX_BOUNCES


class MaterialPreview(Scene):
    def __init__(self) -> None:
        super().__init__()

        pygame.mouse.set_relative_mode(True)

        self.camera.mode = CameraMode.ORBIT
        self.camera.target = pygame.Vector3(0.0)
        self.camera.distance = 5.0
        self.camera.update()

        self.mouse_sensitivity = 0.1
        self.zoom_speed = 50.0

        shared.renderer.settings.color_profile_custom()

        shared.world.clear()
        shared.renderer.update_grid_texture()

        # Enable material preview only experimental feature
        shared.renderer._pt_program["u_exp_raymarch"] = 1

        shared.renderer._pt_program["u_exp_material.albedo"] = (1.0, 1.0, 1.0)
        shared.renderer._pt_program["u_exp_material.emissive"] = (0.0, 0.0, 0.0)
        shared.renderer._pt_program["u_exp_material.metallic"] = 0.0
        shared.renderer._pt_program["u_exp_material.roughness"] = 0.3
        shared.renderer._pt_program["u_exp_material.reflectance"] = 0.04

    def update(self) -> None:
        app = shared.app
        renderer = shared.renderer

        should_reset_acc = False

        for event in app.events:
            if event.type == pygame.MOUSEBUTTONUP:
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
                    renderer.high_quality_snapshot()

            elif event.type == pygame.MOUSEWHEEL:
                scroll = -event.precise_y
                mov = self.zoom_speed * scroll * app.dt
                self.camera.distance += mov
        
        mouse_rel = pygame.Vector2(*pygame.mouse.get_rel())

        if pygame.mouse.get_relative_mode():
            rot = mouse_rel * self.mouse_sensitivity
            self.camera.yaw += rot.x
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

        if should_reset_acc:
            renderer.clear_accumulation()

        self.camera.update()
        self._update_camera_uniform()

    def render(self) -> None:
        app = shared.app
        renderer = shared.renderer

        renderer.render(ui=False)

        imgui.new_frame()
        imgui.begin("Material Preview (press ALT to use UI)", True, flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
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

        if imgui.tree_node("Material", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            _, shared.renderer._pt_program["u_exp_material.albedo"].value = imgui.color_edit3("Albedo", *shared.renderer._pt_program["u_exp_material.albedo"].value, imgui.COLOR_EDIT_NO_INPUTS)
            _, shared.renderer._pt_program["u_exp_material.metallic"] = imgui.slider_float("Metallic", shared.renderer._pt_program["u_exp_material.metallic"].value, 0.0, 1.0, format="%.4f")
            _, shared.renderer._pt_program["u_exp_material.roughness"] = imgui.slider_float("Roughness", shared.renderer._pt_program["u_exp_material.roughness"].value, 0.0, 1.0, format="%.4f")
            _, shared.renderer._pt_program["u_exp_material.reflectance"] = imgui.slider_float("Reflectance (f0)", shared.renderer._pt_program["u_exp_material.reflectance"].value, 0.04, 1.00, format="%.4f")

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

            _, renderer.settings.enable_accumulation = imgui.checkbox("Temporal accumulation", renderer.settings.enable_accumulation)
            _, renderer.settings.nee = imgui.checkbox("Next Event Estimation", renderer.settings.nee)

            aa_name = ("None", "Jitter Sampling", "FXAA")[renderer.settings.antialiasing]
            _, renderer.settings.antialiasing = imgui.slider_int(f"Anti-aliasing", renderer.settings.antialiasing, 0, 2, format=aa_name)

            imgui.tree_pop()

        if imgui.tree_node("Sky", imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_FRAMED):
            _, renderer.settings.enable_sky_texture = imgui.checkbox("Enable sky texture", renderer.settings.enable_sky_texture)
            _, renderer.settings.sky_color = imgui.color_edit3("Sky color", *renderer.settings.sky_color, imgui.COLOR_EDIT_NO_INPUTS)

            _, renderer.settings.sun_angular_radius = imgui.slider_float(f"Sun angular radius", renderer.settings.sun_angular_radius, 0.0, pi)
            _, renderer.settings.sun_yaw = imgui.slider_float(f"Sun yaw", renderer.settings.sun_yaw, 0.0, 360.0)
            _, renderer.settings.sun_pitch = imgui.slider_float(f"Sun pitch", renderer.settings.sun_pitch, -90.0, 90.0)

            imgui.tree_pop()

        if imgui.tree_node("Controls", imgui.TREE_NODE_FRAMED):
            imgui.text(f"[MWHEEL] to zoom in & out")
            imgui.text(f"[R] to reset accumulation")
            imgui.text(f"[ALT] to use toggle cursor and use UI")
            imgui.text(f"[F12] to take high-quality render snapshot")
            imgui.text(f"[ESC] to quit")

            imgui.tree_pop()

        imgui.end()