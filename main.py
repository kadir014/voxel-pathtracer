"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""

from src.app import App
from src.common import *
from src.scenes.sandbox import Sandbox
from src.scenes.material_preview import MaterialPreview


if __name__ == "__main__":
    resolution = (WINDOW_WIDTH, WINDOW_HEIGHT)

    app = App(
        resolution,
        (
            round(resolution[0] * LOGICAL_SCALE),
            round(resolution[1] * LOGICAL_SCALE)
        ),
        target_fps=TARGET_FPS
    )

    app.add_scene(Sandbox)
    #app.add_scene(MaterialPreview)

    app.run()