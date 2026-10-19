import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .common import ROOT, CLASSES


class ManifestDataset(Dataset):
    def __init__(self, split, training=False, image_size=224):
        manifest = ROOT / "data/manifests/candidate_a" / f"{split}.csv"
        with manifest.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.training = training
        ops = [transforms.Resize(256)]
        if training:
            ops += [transforms.RandomResizedCrop(image_size, scale=(.82, 1.0)),
                    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10)]
        else:
            ops += [transforms.CenterCrop(image_size)]
        ops += [transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = ROOT / row["path"]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        label = int(row["label"])
        return image, label, row["image_id"]
