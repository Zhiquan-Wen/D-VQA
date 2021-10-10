import struct
import pathlib

from tempfile import TemporaryFile
from .base import MioBase


class MioWriter(MioBase):
    def __init__(self, root):
        super(MioWriter, self).__init__()

        self.root = pathlib.Path(root)
        self.root.mkdir(exist_ok=True)

        # create 'collections' and 'objects' files
        self.collections = (self.root / "collections").open("xb")
        self.objects = (self.root / "objects").open("xb")

        # temporal file to store the metadata and payload of collections
        self.co_tmp = TemporaryFile("wb+")

        self.collection_indexes = []
        self.collection_id = 0

    def create_collection(self):
        return Colletions(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # write the header info
        self.collections.write(self.magic_number)
        self.collections.write(struct.pack("<HHI", self.major_verion, self.minor_version, len(self.collection_indexes)))

        # re-calculate the collection index offset
        offset = self.header_length + self.collection_serializer.format_.size * len(self.collection_indexes)
        self.collection_indexes = [(start + offset, mlength, plength) for start, mlength, plength in
                                   self.collection_indexes]
        self.collection_serializer.write(self.collections, self.collection_indexes)

        # concat two parts of 'collections'
        self.co_tmp.seek(0)
        for b in iter(lambda: self.co_tmp.read(4096), b""):
            self.collections.write(b)

        # close all the files
        self.co_tmp.close()
        self.objects.close()
        self.collections.close()


class Colletions():
    def __init__(self, m: MioWriter):
        self.m = m
        self.metadata = b""
        self.object_indexes = []
        self.object_id = 0

    @property
    def colletion_id(self):
        return self.m.collection_id

    def set_meta(self, data):
        self.metadata = data

    def add_object(self, bytes_):
        start = self.m.objects.tell()
        length = len(bytes_)
        self.m.objects.write(bytes_)
        self.object_indexes.append((start, length))
        self.object_id += 1

    def close(self):
        metadata_length = len(self.metadata)
        index_len = self.m.object_serializer.format_.size * len(self.object_indexes)
        self.m.collection_indexes.append((self.m.co_tmp.tell(), metadata_length, index_len))
        self.m.co_tmp.write(self.metadata)
        self.m.object_serializer.write(self.m.co_tmp, self.object_indexes)
        self.m.collection_id += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
