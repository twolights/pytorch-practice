import glob
import os.path

import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

PICTURE_SIZE = 96
LABEL_MAPPING = {
    'cat': 0,
    'dog': 1,
}


transform = transforms.Compose([
    transforms.PILToTensor(),
])


def image_to_tensor(picture_path: str) -> torch.Tensor:
    with Image.open(picture_path) as image:
        image.thumbnail((PICTURE_SIZE, PICTURE_SIZE))
        return transform(image) / 255.0


class CatsDogsDataset(Dataset):
    path = None
    is_training = False
    images = []
    labels = []
    data = []

    def _load(self):
        files = glob.glob(self.path + "/*.jpg")
        for f in files:
            filename = os.path.basename(f)
            image = image_to_tensor(f)
            label, _id, _ = filename.split('.')
            self.images.append(image)
            self.labels.append(LABEL_MAPPING[label])

    def __init__(self, path: str, is_training: bool = True):
        self.is_training = is_training
        self.path = path
        files = glob.glob(self.path + "/*.jpg")
        for f in files:
            self.data.append(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        image = image_to_tensor(image_path)
        if self.is_training:
            label, _id, _ = image_path.split('.')
        else:
            label, _id = image_path.split(',')
        return image, label
