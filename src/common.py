"""

    Voxel Path Tracer Project

    This file is a part of the voxel-pathtracer
    project and distributed under MIT license.
    https://github.com/kadir014/voxel-pathtracer

"""


WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

LOGICAL_SCALE = 0.75

# 0 to uncap
TARGET_FPS = 200


# If you don't want your hardware info shown on the UI you can enable this.
# It is mostly for debugging purposes and to see how this project performs
# on different machines with different specs.
HIDE_HW_INFO = False


# These are set so people don't fry their GPU accidentaly
# increase at your own risk

MAX_RAYS_PER_PIXEL = 30
MAX_BOUNCES = 5