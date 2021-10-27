import struct

from io import BytesIO
from PIL import Image

__all__ = [
    "bytes2image",
]


class Serializer(object):
    def __init__(self, format_):
        self.format_ = struct.Struct(format_)

    def read(self, buffer):
        return list(self.format_.iter_unpack(buffer))

    def write(self, f, items):
        for item in items:
            f.write(self.format_.pack(*item))


def bytes2image(b):
    b = BytesIO(b)
    img = Image.open(b)
    return img
