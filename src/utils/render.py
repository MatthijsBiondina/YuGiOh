import sys

import cv2
import numpy as np
import torch

from src.utils.tools import pyout


def show_card(id, version=0):
    fpath = f"res/card_database/images/{id}_{str(version).zfill(2)}.jpg"
    img = cv2.imread(fpath)
    cv2.imshow("Card", img)
    cv2.waitKey(10)


def show_tensor(x):
    h = x.detach().cpu().numpy()
    h -= h.min()
    h /= h.max()

    h = np.transpose(h, axes=(1, 2, 0))
    show(h)


def show(img, waitkey=-1, title="CardMarket"):
    if len(img.shape) == 3:
        cv2.imshow(title, img)
    elif len(img.shape) == 2:
        cv2.imshow(title, img)
    key = cv2.waitKey(waitkey)
    if key == 27:
        sys.exit(0)
