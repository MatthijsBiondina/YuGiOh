import sys
from statistics import median

import cv2
import imutils as imutils
import numpy as np
from imutils.perspective import four_point_transform
from tqdm import tqdm

from src.camera import edgedetection
from src.camera.edgedetection import EdgeDetector
from src.camera.phonecam import PhoneCam
from src.ml.classifier import CardClassifier
from src.utils.render import show
from src.utils.tools import pyout

camera = PhoneCam()
edgedetector = EdgeDetector()
classifier = CardClassifier()

area = []

while True:
    img = camera.get_next_frame()

    orig = img.copy()
    #
    # r = 7
    #
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (r, r), 1)
    # edged = cv2.Canny(gray, 75, 200)
    # # gray = cv2.GaussianBlur(edged, (r, r), 1)
    # # edged = cv2.Canny(gray, 75, 200)
    #
    # # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    # #
    # # frame = np.concatenate((orig, edged), axis=1)
    # # show(frame, waitkey=25)
    #
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    #
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    #
    # screen_cnt = None
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    #
    #     if len(approx) == 4:
    #         screen_cnt = approx
    #         break
    #
    # if screen_cnt is not None:
    #     img = img.copy()
    #     cv2.drawContours(img, [screen_cnt], -1, (0, 255, 0), 2)
    #
    #     # apply the four point transform to obtain a top-down view of the original image
    #     warped = four_point_transform(orig, screen_cnt.reshape(4, 2))
    #     warped = cv2.resize(warped, (421, 614))
    #
    #     img = cv2.resize(img, (421, 614))
    #
    #     frame = np.concatenate([img, warped], axis=1)
    # else:
    #     frame = cv2.resize(orig, (421, 614))
    # show(frame, waitkey=1000 // 64)

    img = edgedetector.process(orig.copy())
    show(img, waitkey=1000 // 64)

    if edgedetector.mode == edgedetection.READY:
        card = edgedetector.get_card(orig.copy())



        card_pred = classifier.classify(card)
        card_set = classifier.release_code(card_pred, card_pred)

        pyout(card_pred.name, card_set)
        card.show(-1)

        pyout()






#
# pyout(median(area))
