import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
from PIL import Image
import pandas as pd
from torchvision import transforms
from tools.utils import nested_tensor_from_tensor_list


def read_data(file):
    info = pd.read_csv(file)
    img_list = info['file_path'].tolist()
    label_list = info['label'].tolist()
    return img_list, label_list


def make_dataset(data_root):
    imgs = os.listdir(data_root)
    imgs = [os.path.join(data_root, img) for img in imgs]
    return imgs


def create_train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
            0.8, 1.5], saturation=[0.2, 1.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def create_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class SeqDeepFakeDataset(Dataset):

    def __init__(self,
                 cfg=None,
                 data_root=None,
                 mode="test",
                 dataset_name=None
                 ):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        if self.mode == "train":
            self.transforms = create_train_transforms(cfg.imgsize)
        elif self.mode in ["val", "test"]:
            self.transforms = create_val_transforms(cfg.imgsize)
        else:
            raise ValueError(f"WRONG INPUT MODE: {self.mode}!")

        # self.data = make_dataset(os.path.join(data_root, f"{dataset_name}/annotations/{mode}.csv"), root=data_root)
        self.data = make_dataset(data_root)

        self.SOS_token_id = cfg.SOS_token_id
        self.EOS_token_id = cfg.EOS_token_id
        self.PAD_token_id = cfg.PAD_token_id

    def __getitem__(self, index: int):

        img_path = self.data[index]
        label_list = [0, 0, 0, 0, 0]

        image = Image.open(img_path).convert('RGB')
        image = image.resize((256, 256))
        if self.transforms:
            image = self.transforms(image)

        return image, torch.FloatTensor(label_list)

    def __len__(self):
        return len(self.data)
