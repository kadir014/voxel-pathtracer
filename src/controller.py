"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

from dataclasses import dataclass

import pygame

from src import shared


@dataclass
class AABB:
    """
    3D axis aligned bounding box.
    """

    min: pygame.Vector3
    max: pygame.Vector3

    def collide(self, aabb: "AABB") -> bool:
        return not (self.max.x < aabb.min.x or aabb.max.x < self.min.x or
                    self.max.y < aabb.min.y or aabb.max.y < self.min.y or
                    self.max.z < aabb.min.z or aabb.max.z < self.min.z)

    def translate(self, position: pygame.Vector3) -> "AABB":
        return AABB(
            self.min + position,
            self.max + position
        )


class PlayerController:
    """
    Player controller.
    """

    def __init__(self, position: pygame.Vector3) -> None:
        self.position = position
        self.velocity = pygame.Vector3(0.0)
        self.movement = pygame.Vector3(0.0)

        gravity = -9.81 * 2.0
        self.gravity = pygame.Vector3(0.0, gravity, 0.0)

        self.horizontal_friction = 0.95

        self.hitbox = AABB(
            pygame.Vector3(0.0, 0.0, 0.0),
            pygame.Vector3(0.5, 1.8, 0.5) * shared.world.voxel_size
        )

        self.on_ground = False

    def movement_per_axis(self, axis: int, dt: float) -> None:
        world = shared.world

        if axis == 1:
            self.on_ground = False

        # Integrate acceleration
        self.velocity[axis] += self.gravity[axis] * dt

        v = self.velocity.copy()
        for i in range(3):
            if i != axis:
                v[i] *= 0

        next_position = self.position + v * dt

        aabb = self.hitbox.translate(next_position)

        for block_coord in world.iter_coords():
            block_id = world[block_coord]
            if (block_id == 0): continue

            block_pos = pygame.Vector3(*block_coord)
            block_pos *= world.voxel_size
            block_aabb = AABB(block_pos, block_pos + pygame.Vector3(world.voxel_size))

            if aabb.collide(block_aabb):
                self.velocity[axis] = 0.0
                if axis == 1:
                    self.on_ground = True
                break

    def update(self, dt: float) -> None:
        world = shared.world

        # Solve Y first to allow gliding on the ground smoothly
        self.movement_per_axis(1, dt)
        self.movement_per_axis(0, dt)
        self.movement_per_axis(2, dt)

        self.velocity.x *= self.horizontal_friction
        self.velocity.z *= self.horizontal_friction

        # Integrate position
        self.position += self.velocity * dt