import json
import os

import numpy as np

from src.core.card import Card
from src.ml.cardmodel import CardArtworkClassifier
from src.ml.image2text import ImageReader
from src.ml.orb import ORB


class CardClassifier:
    ROOT = "res/card_database"

    def __init__(self):
        self.nn_model = CardArtworkClassifier("res/models/resnet.pth")
        self.img2text = ImageReader()
        self.orb_algo = ORB()

        with open(f"{self.ROOT}/ygoprodeck_db.json", 'r') as f:
            self.carddb = json.load(f)['data']
            self.carddb = {card['id']: card for card in self.carddb}
        self.imgfiles = os.listdir(f"{self.ROOT}/images")

    def classify(self, card):
        nnm_ids, nnm_conf = self.nn_model.rank(card)
        txt_ids, txt_conf = self.img2text.rank_title(card)

        if nnm_ids[0] == txt_ids[0]:
            id_ = nnm_ids[0]
        else:
            orb_ids, orb_conf = self.orb_algo.rank(card, np.concatenate((nnm_ids, txt_ids)))
            orb_ids = orb_ids[orb_conf >= 0.98 * orb_conf[0]]
            id_ = txt_ids[0] if txt_ids[0] in orb_ids else orb_ids[0]

        version = self.__get_version(card, id_)

        return Card(self.carddb[id_], version=int(version))

    def release_code(self, card):
        return self.img2text.read_release(card)

    def __get_version(self, card, id_):
        versions = []
        for f in self.imgfiles:
            if int(f.split('_')[0]) == id_:
                versions.append(f.split('_')[-1].split('.')[0])
        versions = sorted(versions)

        if len(versions) == 1:
            return versions[0]

        scores = np.zeros((len(versions),))
        for ii in range(len(versions)):
            path = f"{self.ROOT}/images/{id_}_{versions[ii]}.jpg"
            card_tgt = Card(path)

            scores[ii] = self.orb_algo.compare(card, card_tgt)

        return versions[np.argmax(scores)]
