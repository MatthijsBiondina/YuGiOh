import cv2
import numpy as np
import atexit


class PhoneCam:
    def __init__(self):
        self.cam = cv2.VideoCapture(49)
        atexit.register(cv2.destroyAllWindows)

    def get_next_frame(self):
        N = 0

        ret_val, img = self.cam.read()
        img = np.transpose(img, axes=(1, 0, 2))[::-1, :, :]
        img = img[330 - N:1280 - N]
        return img
