"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

import ast
from time import perf_counter


BLOCK_IDS = (
    None, # 0 is air
    "cobblestone",
    "dirt",
    "glowstone",
    "grass",
    "iron_block",
    "lime_wool",
    "red_wool"
)


class VoxelWorld:
    def __init__(self,
            dimensions: tuple[int, int, int],
            voxel_size: float = 5.0
            ) -> None:
        self.dimensions = dimensions
        self.voxel_size = voxel_size

        self.__map = {}
        for y in range(self.dimensions[1]):
            for z in range(self.dimensions[2]):
                for x in range(self.dimensions[0]):
                    self.__map[(x, y, z)] = 0

        # Grass floor
        for z in range(self.dimensions[2]):
            for x in range(self.dimensions[0]):
                self.__map[(x, 0, z)] = 4

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