import json
import os
import re
from typing import Set

import cv2
import cv2 as cv
import numpy as np
import pytesseract
from difflib import SequenceMatcher

from tqdm import tqdm

from src.core.card import Card
from src.utils.render import show
from src.utils.tools import pyout

escapes = ''.join([chr(char) for char in range(1, 32)])
translator = str.maketrans('', '', escapes)


def read_text_from_img(box):
    txt_nrm = pytesseract.image_to_data(box, output_type='data.frame')

    txt_nrm = txt_nrm[txt_nrm.conf != -1]
    try:
        cnf_nrm = sum(c * len(w) for c, w in zip(txt_nrm.conf, txt_nrm.text)) \
                  / (max(1, sum(len(w) for w in txt_nrm.text)))

        txt_inv = pytesseract.image_to_data(255 - box, output_type='data.frame')
        txt_inv = txt_inv[txt_inv.conf != -1]
        cnf_inv = sum(c * len(str(w)) for c, w in zip(txt_inv.conf, txt_inv.text)) \
                  / (max(1, sum(len(str(w)) for w in txt_inv.text)))

        text = txt_nrm if cnf_nrm > cnf_inv else txt_inv
        text = ' '.join(text.text)
        text = text.translate(translator)

        return text, max(cnf_nrm, cnf_inv) / 100
    except TypeError:
        return "", 0.


def match_sequence(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


class ImageReader:
    def __init__(self):
        with open("res/card_database/ygoprodeck_db.json", 'r') as f:
            data = json.load(f)['data']
        self.cards_by_id = {card['id']: card for card in data}
        self.cards_by_name = {card['name'].lower(): card for card in data}

    def rank_title(self, card):
        text, tesseract_conf = read_text_from_img(card.nameimg)

        if tesseract_conf > 0.9:
            try:
                id_ = self.cards_by_name[text.lower()]['id']
                return np.array([id_]), tesseract_conf
            except KeyError:
                pass

        func = lambda ii: match_sequence(self.cards_by_id[ii]['name'], text.lower())
        rank = sorted(self.cards_by_id.keys(), key=func)[::-1]

        levenshtein_conf = func(rank[0])

        return np.array(rank)[:128], tesseract_conf * levenshtein_conf

    def read_release(self, card: Card, candidates: Set[str]):

        orig = card.releaseimg

        # t = []

        candidates = list(candidates)
        scores = np.zeros((len(candidates),))
        s = [None] * len(candidates)

        height = 13
        for dy in range(0, orig.shape[0] - height, 2):

            img = orig[dy:dy + height]
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            data = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            data = re.sub('[^A-Z0-9-]', '', data)
            # if len(data) > 0:
            #     t.append(data)

            for ii, code in enumerate(candidates):
                s[ii] = match_sequence(code, data)

            idx = max(range(len(s)), key=lambda x: s[x])
            scores[idx] += s[idx]

        idx = max(range(len(candidates)), key=lambda x: scores[x])

        return candidates[idx]

    def read_edition(self, card: Card):
        orig = card.editionimg

        height = 13
        for dy in range(0, orig.shape[0] - height, 2):

            img = orig[dy:dy + height]
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            # show(img, 100)

            data = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
            data = re.sub('[^a-zA-Z0-9-]', '', data)

            score = match_sequence(data, "1st Edition")

            if score > 0.66:
                return True

            # pyout(data, score)

            # for ii, code in enumerate(candidates):
            #     s[ii] = match_sequence(code, data)

            # idx = max(range(len(s)), key=lambda x: s[x])
            # scores[idx] += s[idx]

        # idx = max(range(len(candidates)), key=lambda x: scores[x])

        return False




