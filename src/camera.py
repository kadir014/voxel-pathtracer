"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from math import radians, sin, cos, tan
from enum import Enum, auto

import pygame


class CameraMode(Enum):
    FIRST_PERSON = 0
    ORBIT = 1


class Camera:
    """
    Basic camera setup.

    u and v attributes are the directions on the image plane for shooting rays.

    Attributes
    ----------
    mode
        Camera mode.
    aspect_ratio
        Aspect ratio of the viewport.
    fov
        Field of view in degrees.
    position
        Position of the camera in world space.
        In orbit mode, this is read-only.
    front
        Normalized front vector of the camera in local space.
        This is read-only.
    up
        Normalized up vector of the camera in local space.
    yaw
        Yaw rotation of the camera.
    pitch
        Pitch rotation of the camera.
    target
        Target position of the camera in world space.
        In first person mode, this is read-only.
        In orbit mode, this has to be set manually.
    distance
        Distance from the target.
        This is only used in orbit mode and has to be set manually.
    u
        Horizontal axis of image plane.
    v
        Vertical axis of image plane.
    center
        Center of the image plane.
    """

    def __init__(self,
            viewport: tuple[int, int],
            position: tuple[float, float, float] | pygame.Vector3 = (0.0, 0.0, 0.0),
            fov: float = 50.0,
            mode: CameraMode = CameraMode.FIRST_PERSON
            ) -> None:
        """
        Parameters
        ----------
        viewport
            Dimensions of the viewport.
        position
            Position of the camera in world space.
        fov
            Field of view in degrees.
        mode
            Camera mode.
        """
        self.mode = mode
        self.aspect_ratio = viewport[0] / viewport[1]
        self.fov = fov

        self.position = pygame.Vector3(position)
        self.front = pygame.Vector3(0.0, 0.0, 1.0)
        self.up = pygame.Vector3(0.0, 1.0, 0.0)

        self.yaw = -90.0
        self.pitch = 0.0

        self.target = pygame.Vector3(0.0, 0.0, 0.0)

        # Only for orbit camera
        self.distance = 30.0

        # Image plane attributes
        self.u: pygame.Vector3
        self.v: pygame.Vector3
        self.center: pygame.Vector3

        self.update()

    def copy(self) -> "Camera":
        cam = Camera((1, 1), (0, 0, 0))
        
        cam.mode = self.mode
        cam.aspect_ratio = self.aspect_ratio
        cam.fov = self.fov
        
        cam.position = self.position.copy()
        cam.front = self.front.copy()
        cam.up = self.up.copy()

        cam.yaw = self.yaw
        cam.pitch = self.pitch

        cam.target = self.target.copy()
        cam.distance = self.distance

        cam.u = self.u.copy()
        cam.v = self.v.copy()
        cam.center = self.center.copy()

        return cam

    def update(self) -> None:
        """ Calculate and update camera vectors. """

        self.pitch = pygame.math.clamp(self.pitch, -89.5, 89.5)
        self.fov = pygame.math.clamp(self.fov, 0.0, 180.0)
        self.distance = pygame.math.clamp(self.distance, 0.5, 1000.0)

        pitch_r = radians(self.pitch)
        yaw_r = radians(self.yaw)
        pitch_c = cos(pitch_r)
        pitch_s = sin(pitch_r)
        yaw_c = cos(yaw_r)
        yaw_s = sin(yaw_r)

        # spherical -> cartesian
        sphere = pygame.Vector3(
            yaw_c * pitch_c,
            pitch_s,
            yaw_s * pitch_c
        )

        if self.mode == CameraMode.FIRST_PERSON:
            self.front = sphere.normalize()
            self.target = self.position + self.front

        elif self.mode == CameraMode.ORBIT:
            self.position = self.target + sphere * self.distance
            self.front = (self.target - self.position).normalize()

        alignment = (self.target - self.position).normalize()

        self.u = alignment.cross(self.up).normalize()
        self.v = self.u.cross(alignment).normalize()

        self.center = self.position + alignment

        half_height = tan(radians(self.fov) * 0.5)
        half_width = self.aspect_ratio * half_height

        self.u *= half_width * 2.0
        self.v *= half_height * 2.0

    def move(self, amount: float) -> None:
        """ Move in camera's current direction. """

        self.position += self.front * amount

    def strafe(self, amount: float) -> None:
        """ Strafe horizontally against camera's current direction. """

        right = self.front.cross(self.up).normalize()
        self.position += right * amount