import warnings

from .utils import Serializer


class IllegalFormatException(Exception):
    pass


class IncompatibleException(Exception):
    pass


class MioBase(object):
    def __init__(self):
        self.header_length = 16

        self.magic_number = b"\211tl.mio\032"
        self.major_verion = 0
        self.minor_version = 1

        self.collection_format = "<QII"
        self.collection_serializer = Serializer(self.collection_format)

        self.object_format = "<QI"
        self.object_serializer = Serializer(self.object_format)

    @property
    def version(self):
        return "{}.{}".format(self.major_verion, self.minor_version)

    def _check_magic_number(self, magic_number):
        if not magic_number == self.magic_number:
            raise IllegalFormatException("The directory does not contain legal mio format file.")

    def _check_version(self, major_verion, minor_version):
        if major_verion > self.major_verion:
            raise IncompatibleException("The mio format file version({}.{}) is higher than current one({}.{})."
                                        .format(major_verion, minor_version,
                                                self.major_verion, self.minor_version))
        if minor_version > self.minor_version:
            warnings.warn("The mio format file version({}.{}) is higher than current one({}.{})."
                          .format(major_verion, minor_version,
                                  self.major_verion, self.minor_version))
