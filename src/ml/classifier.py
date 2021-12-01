import json
import os
import sys

import cv2
import numpy as np

from src.core.card import Card
from src.ml.cardmodel import CardArtworkClassifier
from src.ml.image2text import ImageReader
from src.ml.orb import ORB
from src.utils.render import show
from src.utils.tools import pyout


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

        ids = []
        for id_ in np.concatenate((nnm_ids, txt_ids)):
            if self.carddb[id_]['type'] not in ('Skill Card', 'Token'):
                ids.append(id_)
        ids = np.array(ids)


        if nnm_ids[0] == txt_ids[0]:
            id_ = nnm_ids[0]
        else:
            orb_ids, orb_conf = self.orb_algo.rank(card, ids)
            orb_ids = orb_ids[orb_conf >= 0.98 * orb_conf[0]]
            id_ = txt_ids[0] if txt_ids[0] in orb_ids else orb_ids[0]

        version = self.__get_version(card, id_)

        return Card(self.carddb[id_], version=int(version))

    def release_code(self, card_in, card_ou):
        if len(card_ou.card_sets) == 1:
            cardset = card_ou.card_sets[0]
            return [cardset]

        img_pp = self.adjust_hsv_values(card_in, card_ou)

        # show(img_pp)

        # frame = np.concatenate
        candidates = set([set_['set_code'] for set_ in card_ou.card_sets])
        set_code = self.img2text.read_release(Card(img_pp), candidates)

        return [s for s in card_ou.card_sets if s['set_code'] == set_code]

    def edition(self, card_in, card_ou):
        img_pp = self.adjust_hsv_values(card_in, card_ou)
        return self.img2text.read_edition(Card(img_pp))



    def adjust_hsv_values(self, card_in, card_ou):

        hsv_in = cv2.cvtColor(card_in.img, cv2.COLOR_BGR2HSV).astype(int)
        hsv_ou = cv2.cvtColor(card_ou.img, cv2.COLOR_BGR2HSV).astype(int)

        dh1 = hsv_ou[:, :, 0] - hsv_in[:, :, 0]
        dh2 = 180 + hsv_ou[:, :, 0] - hsv_in[:, :, 0]
        dhi = np.where(np.abs(dh1) < np.abs(dh2), dh1, -dh2)

        dh = int(np.median(dhi))
        ds = int(np.median(hsv_ou[:, :, 1] - hsv_in[:, :, 1]))
        dv = int(np.median(hsv_ou[:, :, 2] - hsv_in[:, :, 2]))

        dh, ds, dv = dh, ds, dv

        h = np.mod(hsv_in[:, :, 0] + dh, 180)
        s = np.clip(hsv_in[:, :, 1] + ds, 0, 255)
        v = np.clip(hsv_in[:, :, 2] + dv, 0, 255)

        hsv_pp = np.stack((h, s, v), axis=2).astype(np.uint8)
        img_pp = cv2.cvtColor(hsv_pp, cv2.COLOR_HSV2BGR)

        if card_ou.type == 'XYZ':
            img_pp = 255 - img_pp

        return img_pp

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
