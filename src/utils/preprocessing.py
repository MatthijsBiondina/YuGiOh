from random import randint, uniform

import cv2 as cv
import numpy as np


def preprocess_img(img):
    img = cv.resize(img, (421, 614))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(int)

    # change img
    dh = randint(-10, 10)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + dh, 0, 255)

    ds = randint(-25, 25)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + ds, 0, 255)

    dv = randint(-25, 25)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + dv, 0, 255)

    hsv = hsv.astype(np.uint8)

    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), uniform(-1., 1.), 1.0)
    img = cv.warpAffine(img, M, (w, h))

    return img