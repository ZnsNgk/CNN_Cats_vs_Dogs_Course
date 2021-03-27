import os, glob
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

class data_set(Dataset):
    def __init__(self, folder, transform=None, train=True):
        self.folder = folder
        self.transform = transform
        self.train = train
        img_list = []
        img_list.extend(glob.glob(os.path.join(self.folder,'*.jpg')))
        self.img_list = img_list
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.train:
            if 'dog' in img_path:
                label = 1
            else:
                label = 0
            return img, label
        else:
            (_, img_name) = os.path.split(img_path)
            (name, _) = os.path.splitext(img_name)
            return img, name
    def __len__(self):
        return len(self.img_list)