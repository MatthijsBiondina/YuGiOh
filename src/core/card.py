import cv2 as cv
import numpy as np

from src.utils.render import show
from src.utils.tools import pyout


class Card:
    ROOT = "res/card_database"

    def __init__(self, arg, version=0):
        self.id = None
        self.name = None
        self.card_sets = None
        if type(arg) == str:
            self.img = cv.resize(cv.imread(arg), (421, 614))
        elif type(arg) == np.ndarray:
            self.img = arg
        elif type(arg) == dict:
            self.__init_from_dict(arg, version)
        else:
            raise TypeError(f"{type(arg)} not supported, should be one of (str, dict, ndarray)")

    @property
    def nameimg(self):
        return self.img[29:87, 27:351]
        # return self.img[29:67, 27:351]

    @property
    def artwork(self):
        return self.img[108:458, 26:395]

    @property
    def releaseimg(self):
        return self.img[:, :]

    def show(self, waitkey=-1):
        show(self.img, waitkey=waitkey)

    def __init_from_dict(self, D, v=0):
        path = f"{self.ROOT}/images/{D['id']}_{str(v).zfill(2)}.jpg"
        self.img = cv.resize(cv.imread(path), (421, 614))
        self.id = D['id']
        self.name = D['name']
        self.card_sets = D['card_sets']
