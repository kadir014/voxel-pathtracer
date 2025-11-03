"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

from src import shared
from src.camera import Camera


class Scene:
    def __init__(self) -> None:
        self.camera = Camera(shared.app._logical_resolution)
        self.prev_camera = self.camera.copy()

    def _update_camera_uniform(self) -> None:
        shared.renderer._pt_program["u_camera.position"] = (
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        )
        shared.renderer._pt_program["u_camera.center"] = (
            self.camera.center.x,
            self.camera.center.y,
            self.camera.center.z
        )
        shared.renderer._pt_program["u_camera.u"] = (
            self.camera.u.x,
            self.camera.u.y,
            self.camera.u.z
        )
        shared.renderer._pt_program["u_camera.v"] = (
            self.camera.v.x,
            self.camera.v.y,
            self.camera.v.z
        )

        shared.renderer._pt_program["u_prev_camera.position"] = (
            self.prev_camera.position.x,
            self.prev_camera.position.y,
            self.prev_camera.position.z
        )
        shared.renderer._pt_program["u_prev_camera.center"] = (
            self.prev_camera.center.x,
            self.prev_camera.center.y,
            self.prev_camera.center.z
        )
        shared.renderer._pt_program["u_prev_camera.u"] = (
            self.prev_camera.u.x,
            self.prev_camera.u.y,
            self.prev_camera.u.z
        )
        shared.renderer._pt_program["u_prev_camera.v"] = (
            self.prev_camera.v.x,
            self.prev_camera.v.y,
            self.prev_camera.v.z
        )

    def update(self) -> None:
        """ Called each frame for updating. """

    def render(self) -> None:
        """ Called each frame for rendering. """