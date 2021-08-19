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

    img = edgedetector.process(orig.copy())
    show(img, waitkey=1000 // 64)

    if edgedetector.mode == edgedetection.READY:
        card_in = edgedetector.get_card(orig.copy())

        card_ou = classifier.classify(card_in)
        card_set = classifier.release_code(card_in, card_ou)
        edition_1st = classifier.edition(card_in, card_ou)

        pyout(card_ou.name, card_set[0]['set_code'])
        card_ou.show(1000)








#
# pyout(median(area))
