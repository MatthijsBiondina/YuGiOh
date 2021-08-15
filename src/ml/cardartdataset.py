import os
from random import randint

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms

from src.utils.tools import pyout


class CardArtTrainSet(Dataset):
    def __init__(self):
        super(CardArtTrainSet, self).__init__()
        root = "res/card_database/images"
        self.img_paths = [f"{root}/{img}" for img in os.listdir(root)]
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((244, 244)),
            transforms.ColorJitter(brightness=(0.5, 1.5),
                                   contrast=(0.3, 2.0), hue=.05,
                                   saturation=(.5, 1.5)),
            transforms.RandomAffine(0, translate=(0, 0.3),
                                    scale=(0.6, 1.8),
                                    shear=(0.0, 0.4),
                                    fill=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = image.resize((421, 614), Image.ANTIALIAS)


        rshift = lambda x: x + randint(-5, 5)

        img_anc = image.crop((rshift(26), rshift(108), rshift(395), rshift(458)))
        img_pos = image.crop((rshift(26), rshift(108), rshift(395), rshift(458)))

        img_anc = self.transform(img_anc)
        img_pos = self.transform(img_pos)

        return img_anc, img_pos


class CardArtEvalSet(Dataset):
    def __init__(self, labels=False):
        super(CardArtEvalSet, self).__init__()
        self.labels = labels
        root = "res/card_database/images"
        self.img_paths = [f"{root}/{img}" for img in os.listdir(root)]
        self.transform_anchor = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform_target = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ColorJitter(brightness=(0.8, 1.2),
                                   contrast=(0.8, 1.2),
                                   hue=.05,
                                   saturation=(0.9, 1.1)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(0, translate=(.1, 0.1), scale=(0.95, 1.05),
                                    shear=(0.0, 0.05), fill=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        image = image.resize((421, 614), Image.ANTIALIAS)
        rshift = lambda x: x + randint(-2, 2)

        img_anc = image.crop((26, 108, 395, 458))
        img_tgt = image.crop((rshift(26), rshift(108), rshift(395), rshift(458)))

        img_anc = self.transform_anchor(img_anc)
        img_tgt = self.transform_target(img_tgt)

        if self.labels:
            id = int(self.img_paths[idx].split('/')[-1].split('_')[0])
            return img_tgt, id
        else:
            return img_anc, img_tgt
