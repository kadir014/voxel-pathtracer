"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""


__version__ = "0.0.3"


# Window resolution
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# Renderer logical resolution
# e.g a logical scale of 0.5 would render at half the window resolution
LOGICAL_SCALE = 0.75

# 0 to uncap
TARGET_FPS = 200

# If you don't want your hardware info shown on the UI you can enable this.
# It is mostly for debugging purposes and to see how this project performs
# on different machines with different specs.
HIDE_HW_INFO = False


# These are set so people don't fry their GPU accidentaly
# increase at your own risk

MAX_RAYS_PER_PIXEL = 2 ** 8
MAX_BOUNCES = 5