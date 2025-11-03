"""

    Project Lyrae | Physically-based real-time voxel graphics

    This file is a part of the Lyrae Project
    and distributed under MIT license.
    https://github.com/kadir014/project-lyrae

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.app import App
    from src.world import VoxelWorld
    from src.renderer import Renderer


app: "App"

world: "VoxelWorld"

renderer: "Renderer"