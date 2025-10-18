"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

import pygame


class Camera:
    """
    Simple first-person camera.

    A simple camera model using a physical camera setup defined 
    by lens focal length and image horizontal size.

    u and v attributes are the directions on the image plane for shooting rays.
    """

    def __init__(self,
            viewport: tuple[int, int],
            position: tuple[float, float, float] | pygame.Vector3 = (0.0, 0.0, 0.0)
            ) -> None:
        self.aspect_ratio = viewport[0] / viewport[1]
        self.position = pygame.Vector3(position)
        self.look_at = pygame.Vector3(0.0, 0.0, 1.0)
        self.up = pygame.Vector3(0.0, 1.0, 0.0)

        self.focal_length = 0.15
        self.horizontal_size = 0.16

        self.update()

    def update(self) -> None:
        """ Calculate and update camera vectors. """

        alignment = (self.look_at - self.position).normalize()

        self.u = alignment.cross(self.up).normalize()
        self.v = self.u.cross(alignment).normalize()

        self.center = self.position + self.focal_length * alignment

        self.u *= self.horizontal_size
        self.v *= self.horizontal_size / self.aspect_ratio

    def move(self, amount: float) -> None:
        """ Move in camera's current direction. """

        delta = (self.look_at - self.position).normalize() * amount
        self.position += delta
        self.look_at += delta

    def strafe(self, amount: float) -> None:
        """ Strafe horizontally. """

        delta = (self.look_at - self.position).normalize()
        delta = delta.cross(pygame.Vector3(0.0, 1.0, 0.0)) * amount
        self.position += delta
        self.look_at += delta

    def rotate_yaw(self, amount: float) -> None:
        """ Rotate camera horizontally. """

        delta = self.look_at - self.position
        delta = delta.rotate(amount, self.up)
        self.look_at = self.position + delta

    def rotate_pitch(self, amount: float) -> None:
        """ Rotate camera vertically. """

        delta = self.look_at - self.position
        delta = delta.rotate(amount, self.up.cross(delta.normalize()))
        self.look_at = self.position + delta