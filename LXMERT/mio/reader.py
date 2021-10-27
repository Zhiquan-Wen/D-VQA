import mmap
import pathlib
import struct

from .base import MioBase


class MioReader(MioBase):
    def __init__(self, root):
        super(MioReader, self).__init__()
        self.root = pathlib.Path(root)
        self.collections = (self.root / "collections").open("rb")
        self.objects = (self.root / "objects").open("rb")

        # check magic number
        magic_number = self.collections.read(8)
        self._check_magic_number(magic_number)

        # check major and minor version
        major_verion, minor_version = struct.unpack("<HH", self.collections.read(4))
        self._check_version(major_verion, minor_version)

        self.n_collections = struct.unpack("<I", self.collections.read(4))[0]

        # mmap
        self.collections_map = mmap.mmap(
            fileno=self.collections.fileno(),
            length=0,
            access=mmap.ACCESS_READ
        )
        self.objects_map = mmap.mmap(
            fileno=self.objects.fileno(),
            length=0,
            access=mmap.ACCESS_READ
        )
        collection_index_start = self.header_length
        collection_index_end = collection_index_start + self.collection_serializer.format_.size * self.size
        self.collections_indexes = self.collection_serializer.read(
            self.collections_map[collection_index_start:collection_index_end])

    @property
    def size(self) -> int:
        return self.n_collections

    def get_collection_metadata(self, collection_id):
        start, metadata_length, payload_length = self._get_collection(collection_id)
        end = start + metadata_length
        return self.collections_map[start:end]

    def get_collection_size(self, collection_id):
        start, metadata_length, payload_length = self._get_collection(collection_id)
        return payload_length // self.object_serializer.format_.size

    def fetchone(self, colletion_id, object_id=0):
        return self._fetch(colletion_id, object_id)

    def fetchmany(self, colletion_id, object_ids):
        return self._fetch(colletion_id, object_ids)

    def fetchall(self, colletion_id):
        return self._fetch(colletion_id, None)

    def close(self):
        self.objects_map.close()
        self.objects.close()

        self.collections_map.close()
        self.collections.close()

    def _fetch(self, colletion_id, object_ids):
        object_indexes = self._get_collection_payload(colletion_id)
        if isinstance(object_ids, int):
            object_id = object_ids
            start, length = object_indexes[object_id]
            end = start + length
            return self.objects_map[start:end]
        if object_ids is not None:
            object_indexes = [object_indexes[i] for i in object_ids]
        return [self.objects_map[start:start + length] for start, length in object_indexes]

    def _get_collection(self, collection_id):
        collection_index = self.collections_indexes[collection_id]
        return collection_index

    def _get_collection_payload(self, collection_id):
        start, metadata_length, payload_length = self._get_collection(collection_id)
        payload_start = start + metadata_length
        payload_end = payload_start + payload_length
        payload = self.object_serializer.read(self.collections_map[payload_start:payload_end])
        return payload
