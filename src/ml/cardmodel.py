import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
import antialiased_cnns
from torchvision.transforms import transforms
from tqdm import tqdm

from src.core.card import Card


class CardModel(nn.Module):
    def __init__(self, architecture="resnet"):
        super(CardModel, self).__init__()
        if architecture == "resnet":
            self.model = antialiased_cnns.resnet101(pretrained=True)
            # self.model = models.mobilenet_v3_small(pretrained=True)
        elif architecture == "mobile":
            self.model = models.mobilenet_v3_small(pretrained=True)
        else:
            raise ValueError(f"Pretrained Architecture {architecture} not available.")

        # self.mobile = antialiased_cnns.mobilenet_v2(pretrained=True)

    def forward(self, x):
        h = self.model(x)

        return h


class CardArtworkClassifier:
    def __init__(self, state_dict):
        self.model = CardModel()
        self.model.load_state_dict(torch.load(state_dict))
        self.model = self.model.cuda()
        self.model.eval()
        self.idx2id = np.load(f"res/card_database/orb.npz")['card_id']

        self.transform = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        try:
            self.feature_vectors = np.load("res/card_database/feature_vectors.npy")
        except FileNotFoundError:
            self.feature_vectors = self.__init_feature_vectors()

    def compute_feature_vector(self, card):
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(card.artwork, cv2.COLOR_BGR2RGB))
            X = self.transform(img).unsqueeze(0).cuda()
            return self.model(X).squeeze(0)

    def rank(self, card):
        fv = self.compute_feature_vector(card).cpu().numpy()

        dist = ((self.feature_vectors - fv[None, :]) ** 2).sum(axis=1) ** .5
        idx = np.argsort(dist)
        id_ = self.idx2id[idx]
        return id_[:128], dist[idx[0]]

    def __init_feature_vectors(self):
        imgfiles = sorted(os.listdir(f"res/card_database/images"))
        FV = np.zeros((len(imgfiles), 1000))

        for ii, file in enumerate(tqdm(imgfiles)):
            path = f"res/card_database/images/{file}"
            card = Card(path)
            with torch.no_grad():
                fv = self.compute_feature_vector(card)

            fv = fv.cpu().numpy()
            FV[ii] = fv

        np.save("res/card_database/feature_vectors.npy", FV)
        return FV
