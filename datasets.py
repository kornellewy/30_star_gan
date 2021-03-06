import glob
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebADataset(Dataset):
    def __init__(
        self, root, label_path=None, transforms_=None, mode="train", attributes=None
    ):
        self.transform = transforms.Compose(transforms_)
        self.selected_attrs = attributes
        self.files = sorted(glob.glob("%s\\*.jpg" % root))
        valid_size = len(self.files) // 10
        self.files = self.files if mode == "train" else self.files[-valid_size:]
        if not label_path:
            self.label_path = glob.glob("%s\\*.txt" % root)[0]
        self.annotations = self.get_annotations()

    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[0].split()
        for _, line in enumerate(lines[1:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("\\")[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))
        return img, label

    def __len__(self):
        return len(self.files)
