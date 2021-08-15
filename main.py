import json
from random import randint, randrange, uniform

import cv2 as cv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.camera.card import Card
from src.ml.cardartdataset import CardArtEvalSet
from src.ml.cardmodel import CardModel, CardArtworkClassifier
from src.ml.classifier import CardClassifier
from src.ml.image2text import ImageReader
from src.ml.orb import ORB
from src.utils.preprocessing import preprocess_img
from src.utils.render import show, show_tensor
from src.utils.tools import pyout

classifier = CardClassifier()

ROOT = "res/card_database"
imgfiles = sorted(os.listdir(f"{ROOT}/images"))

for ii, fname in enumerate(tqdm(imgfiles)):
    id_ = int(fname.split('_')[0])

    fname = f"{ROOT}/images/{fname}"
    img_ = cv.imread(fname)
    img_ = preprocess_img(img_)
    card = Card(img_)

    card_pred = classifier.classify(card)

    nn_ids, nn_conf = nn.rank(card)
    txt_ids, txt_conf = read.rank_title(card)
    if nn_ids[0] == txt_ids[0]:
        ml_id = nn_ids[0]
    else:
        orb_ids, orb_conf = orb.rank(card, np.concatenate((nn_ids[:128], txt_ids[:128])))
        orb_ids = orb_ids[orb_conf >= 0.99 * orb_conf[0]]
        if txt_ids[0] in orb_ids:
            ml_id = txt_ids[0]
        else:
            ml_id = orb_ids[0]

    if ml_id != id_:
        pyout(carddb[id_]['name'])

        frame = [card.artwork]
        for ii in range(len(carddb[ml_id]['card_images'])):
            img_ = Card(f"res/card_database/images/{ml_id}_{str(ii).zfill(2)}.jpg")
            frame.append(img_.artwork)

        show(np.concatenate(frame, axis=1), waitkey=100)
