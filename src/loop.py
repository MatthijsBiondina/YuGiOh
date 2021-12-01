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
from src.gui import GUI
from src.ml.classifier import CardClassifier
from src.scraping.cardmarket import scrape_card_mint_price
from src.utils.render import show
from src.utils.tools import pyout


def loop():
    camera = PhoneCam()
    edgedetector = EdgeDetector()
    classifier = CardClassifier()

    gui = GUI()

    area = []

    running = True
    total_price = 0.

    try:
        with open("res/store", "r") as f:
            for line in f:
                total_price += float(line.split(",")[2])
    except FileNotFoundError:
        pass

    while running:
        img = camera.get_next_frame()
        orig = img.copy()
        img = edgedetector.process(orig.copy())

        gui.set_camera_img(img.copy())

        running = gui.update()

        if edgedetector.mode == edgedetection.READY:
            card_in = edgedetector.get_card(orig.copy())

            gui.set_camera_img(card_in.img)
            gui.update()

            card_ou = classifier.classify(card_in)

            gui.set_process_img(card_ou.img)
            gui.set_card_name(card_ou.name)
            gui.update()

            card_set = classifier.release_code(card_in, card_ou)

            done = False
            while not done:

                gui.set_card_set(card_set)
                gui.update()

                price = scrape_card_mint_price(card_ou.name, card_set[0])
                gui.set_price(price)
                gui.update()

                quality = gui.query_quality(card_ou)

                if isinstance(quality, str):
                    for s in card_ou.card_sets:
                        if s['set_code'] == quality:
                            card_set = [s]
                            break
                    gui.set_card_set(None)
                    gui.set_price(None)
                elif quality:
                    quality_str, quality_price = quality
                    total_price += quality_price
                    gui.set_total_price(total_price)
                    with open("res/store", "a+") as f:
                        f.write(f"{card_ou.id},{card_set[0]['set_code']},{price},{quality_str}\n")
                    done = True
                else:
                    done = True

            gui.all_card_sets = None
            gui.set_process_img(None)
            gui.set_card_name(None)
            gui.set_card_set(None)

#
# pyout(median(area))
