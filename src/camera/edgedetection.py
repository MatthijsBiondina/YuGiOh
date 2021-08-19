import time

import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from numpy import pi
import matplotlib.cm as cm

from src.core.card import Card
from src.utils.render import show
from src.utils.tools import pyout

WAITING = 0
DETECTING = 1
READY = 2
COOLDOWN = 3


class EdgeDetector:
    N = 64
    TOL = 0.08
    ELENS = np.array([750, 518, 750, 518])

    def __init__(self):
        self.mode = WAITING
        self.ii = 0

        self.valid = np.zeros((self.N,), dtype=bool)
        self.cntrs = np.zeros((self.N, 4, 2), dtype=int)

    def reset_arrays(self):
        self.valid = np.zeros((self.N,), dtype=bool)

    def process(self, img):
        cntr_ = self.__get_contours(img.copy())
        if cntr_ is not None:
            cntr_ = self.sort_cnts(cntr_)
            self.valid[self.ii] = self.__check_valid(cntr_)

            if self.mode == WAITING:
                ou_img = self._wait(img, cntr_)
            elif self.mode == DETECTING:
                ou_img = self._detect(img, cntr_)
            elif self.mode == READY:
                pyout("Edge-detector is ready.")
                ou_img = img
            elif self.mode == COOLDOWN:
                ou_img = self._cooldown(img, cntr_)
        else:
            self.valid[self.ii] = False
            if self.mode == COOLDOWN:
                ou_img = self._cooldown(img, cntr_)
            else:
                ou_img = img

        self.ii = (self.ii + 1) % self.N

        return ou_img

    def get_card(self, img):
        mu = np.mean(self.cntrs[self.valid], axis=0).astype(int)

        warped = four_point_transform(img, mu.reshape(4, 2))
        warped = cv2.resize(warped, (421, 614), interpolation=cv2.INTER_AREA)

        self.mode = COOLDOWN

        return Card(warped)

    def _wait(self, img, c):
        n_valid = np.sum(self.valid)
        prog = min(n_valid / (0.5 * self.N), 1.)

        if prog == 1.:
            self.mode = DETECTING
            self.reset_arrays()

        color = self.color(prog, cm.spring)
        for jj in range(4):
            pt1 = c[jj]
            pt2 = c[(jj + 1) % 4]
            img = cv2.line(img, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)

        return img

    def _detect(self, img, c):
        n_valid = np.sum(self.valid)
        prog = min(n_valid / (0.5 * self.N), 1.)

        if prog == 1.:
            self.mode = READY

        if self.valid[self.ii]:
            self.cntrs[self.ii] = c

        mu = np.mean(self.cntrs[self.valid], axis=0).astype(int)

        color = self.color(1. - prog, cm.summer)

        for jj in range(4):
            pt1 = c[jj]
            pt2 = c[(jj + 1) % 4]
            img = cv2.line(img, pt1, pt2, color, thickness=3, lineType=cv2.LINE_AA)

        return img

    def _cooldown(self, img, c):
        n_valid = np.sum(self.valid)
        if n_valid == 0:
            self.mode = WAITING

        prog = min(n_valid / (0.5 * self.N), 1.)
        color = self.color(prog, cm.winter)
        cv2.rectangle(img, (0, 0), img.shape[:2][::-1], color=color, thickness=5)

        return img

    def __get_contours(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # show(edged)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is None:
            return None
        return screen_cnt.squeeze(1)

    def __check_valid(self, cnts):
        lens = ((cnts - np.roll(cnts, 1, axis=0)) ** 2).sum(axis=1) ** .5
        lens = (lens > self.ELENS * (1 - self.TOL)) & (lens < self.ELENS * (1 + self.TOL))

        lens_ok = np.all(lens)

        area = cv2.contourArea(cnts[:, None, :]) ** .5
        area_tgt = (self.ELENS[0] * self.ELENS[1]) ** .5

        area_ok = (area > area_tgt * (1 - self.TOL)) & (area < area_tgt * (1 + self.TOL))

        return lens_ok and area_ok

    def sort_cnts(self, cnts):
        orig = np.mean(cnts, axis=0)
        ang = np.arctan2(cnts[:, 1] - orig[None, 1], cnts[:, 0] - orig[None, 0]) % (2 * pi)
        cnts = cnts[np.argsort(ang)]
        return cnts

    def color(self, prog, cmap=cm.spring):
        alpha = 0.8

        c = np.array(cmap(prog)[:3][::-1]) * 255
        c = (alpha * c + (1 - alpha) * 255)
        return c.astype(np.uint8).tolist()
