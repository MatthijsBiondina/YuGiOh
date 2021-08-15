import json
import os

import cv2 as cv
import numpy as np
import pytesseract
from difflib import SequenceMatcher

from tqdm import tqdm

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

# with open("res/card_database/ygoprodeck_db.json", 'r') as f:
#     data = json.load(f)['data']
#     cards_by_id = {card['id']: card for card in data}
#     cards_by_name = {card['name'].lower(): card for card in data}
#
# root = "res/card_database/images"
# for jj, fname in enumerate(tqdm(os.listdir(root))):
#     fname = f"{root}/{fname}"
#
#     img_ = cv.imread(fname, 0)
#     img_ = cv.resize(img_, (421, 614))
#     img = img_[29:67, 27:351]
#
#     text, tesseract_conf = read_text_from_img(img)
#
#     try:
#         found_card = cards_by_name[text.lower()]
#         levenshtein_conf = 100
#     except KeyError:
#         f = lambda ii: SequenceMatcher(None, data[ii]['name'].lower(), text.lower()).ratio()
#         card_argmax = max(range(len(data)), key=f)
#         found_card = data[card_argmax]
#         levenshtein_conf = round(f(card_argmax) * 100)
#
#     real_card = cards_by_id[int(fname.split('/')[-1].split('_')[0])]
#
#     if found_card['id'] != real_card['id']:
#         tqdm.write("\n")
#         tqdm.write(f"READ:  {text}")
#         tqdm.write(f"FOUND: {found_card['name']}")
#         tqdm.write(f"REAL:  {real_card['name']}")
#         tqdm.write(f"tesseract:   {int(tesseract_conf)}%")
#         tqdm.write(f"levenshtein: {int(levenshtein_conf)}%")
#         img_ = cv.imread(fname)
#         show(img_[:, :, ::-1], 100)
