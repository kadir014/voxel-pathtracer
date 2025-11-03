"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

import ast
from time import perf_counter
from math import floor
from dataclasses import dataclass

import pygame


BLOCK_IDS = (
    None, # 0 is air
    "cobblestone",
    "dirt",
    "glowstone",
    "grass",
    "iron_block",
    "lime_wool",
    "red_wool",
    "red_light",
    "copper_block"
)


@dataclass
class HitInfo:
    """
    Raycast hit information.

    Attributes
    ----------
    hit
        Collision with ray happened?
    point
        Collision point on the surface in world space
    normal
        Normal of the collision surface
    voxel
        Voxel coordinate in grid space
    """
    hit: bool
    point: pygame.Vector3
    normal: pygame.Vector3
    voxel: tuple[int, int, int]


class VoxelWorld:
    def __init__(self,
            dimensions: tuple[int, int, int],
            voxel_size: float = 5.0
            ) -> None:
        self.dimensions = dimensions
        self.voxel_size = voxel_size

        self.__map = {}
        self.clear()

        # Grass floor
        for z in range(self.dimensions[2]):
            for x in range(self.dimensions[0]):
                self.__map[(x, 0, z)] = BLOCK_IDS.index("grass")

    def __getitem__(self, key: tuple[int, int, int]) -> int:
        return self.__map[key]
    
    def __setitem__(self, key: tuple[int, int, int], value: int) -> None:
        self.__map[key] = value

    def __contains__(self, key: tuple[int, int, int]) -> bool:
        return key in self.__map
    
    def save(self, filepath: str) -> None:
        start = perf_counter()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(repr(self.__map))

        elapsed = perf_counter() - start
        print(f"Saved map in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")

    def load(self, filepath: str) -> None:
        start = perf_counter()

        with open(filepath, "r", encoding="utf-8") as f:
            self.__map = ast.literal_eval(f.read())

        elapsed = perf_counter() - start
        print(f"Loaded map in {round(elapsed, 3)}s ({round(elapsed*1000.0, 3)}ms)")

    def clear(self) -> None:
        for y in range(self.dimensions[1]):
            for z in range(self.dimensions[2]):
                for x in range(self.dimensions[0]):
                    self.__map[(x, y, z)] = 0

    def dda(self,
            origin: pygame.Vector3,
            dir: pygame.Vector3,
            max_steps: int = 256
            ) -> HitInfo:
        """
        Cast a ray into the world and collect hit information using DDA.

        Parameters
        ----------
        origin
            Origin of the ray.
        dir
            Direction of the ray.
        max_steps
            Maximum traversal steps before terminating. 
        """
        voxel = origin / self.voxel_size
        voxel.x = floor(voxel.x)
        voxel.y = floor(voxel.y)
        voxel.z = floor(voxel.z)

        step = pygame.Vector3(
            int(dir.x > 0) - int(dir.x < 0),
            int(dir.y > 0) - int(dir.y < 0),
            int(dir.z > 0) - int(dir.z < 0)
        )

        next_boundary = pygame.Vector3(
            (voxel.x + (1.0 if step.x > 0.0 else 0.0)) * self.voxel_size,
            (voxel.y + (1.0 if step.y > 0.0 else 0.0)) * self.voxel_size,
            (voxel.z + (1.0 if step.z > 0.0 else 0.0)) * self.voxel_size
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
        else: t_delta.x = abs(self.voxel_size / dir.x)
        if dir.y == 0.0: t_delta.y = float("inf")
        else: t_delta.y = abs(self.voxel_size / dir.y)
        if dir.z == 0.0: t_delta.z = float("inf")
        else: t_delta.z = abs(self.voxel_size / dir.z)

        normal = pygame.Vector3(0.0)

        # traverse
        for i in range(max_steps):
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
            if ivoxel not in self.__map: continue
            sample = self.__map[ivoxel]
            if sample > 0:
                return HitInfo(
                    True,
                    origin + dir * hit_t,
                    normal,
                    ivoxel
                )
            
        return HitInfo(False, pygame.Vector3(), pygame.Vector3(), (0, 0, 0))