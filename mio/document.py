import os
import mmap
from .record import Record


class Document(object):
    def __init__(self, filename, readonly=True):
        self.readonly = readonly
        if readonly:
            self.file = open(filename, "rb")
            self.mmap = mmap.mmap(
                fileno=self.file.fileno(),
                length=0,
                access=mmap.ACCESS_READ
            )
        else:
            if os.path.exists(filename):
                self.file = open(filename, "ab")
            else:
                self.file = open(filename, "wb")

    def put(self, item, pipelines=None) -> Record:
        if pipelines is not None:
            for p in pipelines:
                item = p(item)
        start = self.file.tell()
        length = len(item)
        self.file.write(item)
        return Record(start,length)

    def get(self, record: Record, pipelines=None):
        item = memoryview(self.mmap[record.start:record.end])
        if pipelines is not None:
            for p in pipelines:
                item = p(item)
        return item

    def release(self):
        if self.readonly:
            self.mmap.close()
        self.file.close()
