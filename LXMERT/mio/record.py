class Record(object):
    def __init__(self, start, length):
        self.start = start
        self.length = length

    @property
    def end(self):
        return self.start + self.length

    def __str__(self):
        return "Record({},{})".format(self.start, self.end)
