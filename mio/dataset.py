import copy

from torch.utils.data import Dataset
from .reader import MioReader
from .split import Split

class Subset(Dataset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        if transform is not None:
            self.dataset.transform = transform
        if target_transform is not None:
            self.dataset.target_transform = target_transform

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class MioDataset(Dataset):
    def __init__(self, root, sampler, transform=None, target_transform=None):
        self.root = root

        self.sampler = sampler

        self.transform = transform
        self.target_transform = target_transform

        self.mio = MioReader(self.root)

    def to_split(self, split: Split, transform=None, target_transform=None):
        return Subset(self, split.items, transform, target_transform)

    def __getitem__(self, id_):
        size = self.mio.get_collection_size(id_)
        selected_samples = self.sampler(id_, size)

        if isinstance(selected_samples, int):
            object_id = selected_samples
            data = self.mio.fetchone(id_, object_id)
        else:
            data = self.mio.fetchmany(id_, selected_samples)

        target = self.mio.get_collection_metadata(id_)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.mio.size
