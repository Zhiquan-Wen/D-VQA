import pathlib
import struct


class Split(object):
    def __init__(self, items):
        self.items = items
        assert isinstance(items, list)
        if not self._is_unique():
            raise Exception("The items contains repeating item.")
        self.items = sorted(self.items)
        self.serializer = struct.Struct("<I")

    def _is_unique(self):
        return len(self.items) == len(set(self.items))

    def tofile(self, path):
        with pathlib.Path(path).open("xb") as f:
            for item in self.items:
                f.write(self.serializer.pack(item))

    @staticmethod
    def fromfile(path):
        serializer = struct.Struct("<I")
        with pathlib.Path(path).open("rb") as f:
            b = f.read()
            items = serializer.iter_unpack(b)
        return Split([item[0] for item in items])
