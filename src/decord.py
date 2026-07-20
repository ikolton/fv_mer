"""Minimal import surface required by LAVIS for non-video CT workflows."""


class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Video decoding is not available in this CT environment")


def cpu(*args, **kwargs):
    return None


class _Bridge:
    def set_bridge(self, name):
        return None


bridge = _Bridge()

