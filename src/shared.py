"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app import App
    from src.world import VoxelWorld
    from src.renderer import Renderer


app: "App"

world: "VoxelWorld"

renderer: "Renderer"