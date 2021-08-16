import os
from math import cos, sin

import numpy as np
import cupy as cp
import cv2 as cv
from numpy import radians
from tqdm import tqdm

from src.core.card import Card


class ORB:
    ROOT = "res/card_database"
    BS = 64
    N_FEATURES = 100

    def __init__(self, path=ROOT):
        self.orb_algorithm = cv.ORB_create(nfeatures=self.N_FEATURES)
        try:
            npzfile = np.load(f"{path}/orb.npz")
            self.keypoints = npzfile['orb_clouds']
            self.idx2id = npzfile['card_id']
            self.id2idx = {}
            for ii, id in enumerate(self.idx2id):
                try:
                    self.id2idx[id].append(ii)
                except KeyError:
                    self.id2idx[id] = [ii]

        except (FileNotFoundError, TypeError):
            self.keypoints, self.idx2id = self.__init_orb_clouds()

    def compare(self, card1, card2):
        artwork1 = cv.cvtColor(card1.artwork, cv.COLOR_BGR2GRAY)
        artwork2 = cv.cvtColor(card2.artwork, cv.COLOR_BGR2GRAY)

        h, w = artwork1.shape[:2]
        kp1 = self.orb_algorithm.detect(artwork1, None)
        kp2 = self.orb_algorithm.detect(artwork2, None)
        pc1 = np.stack([self.kp2vec(kp, w, h) for kp in kp1])
        pc2 = np.stack([self.kp2vec(kp, w, h) for kp in kp2])

        M = ((pc1[None, :, :2] - pc2[:, None, :2]) ** 2).sum(-1) ** .5
        D = (pc1[None, :, 2:4] * pc2[:, None, 2:4]).sum(-1)
        S = (1 - M) * D

        score = np.mean(np.max(S, axis=1), axis=0)

        return score

    def rank(self, card, ids=None):
        img = cv.cvtColor(card.artwork, cv.COLOR_BGR2GRAY)
        w, h = img.shape[:2]
        kp = self.orb_algorithm.detect(img, None)
        kp = np.stack([self.kp2vec(p, w, h) for p in kp])
        kp = cp.asarray(kp)

        if ids is None:
            ids = self.idx2id

        idxs = []
        for id in ids:
            idxs.extend(self.id2idx[id])
        idxs = np.array(list(set(idxs)))

        similarity = np.zeros(idxs.shape[0])

        for ii_frm in range(0, ids.shape[0] + self.BS, self.BS):
            ii_to = min(ii_frm + self.BS, similarity.shape[0])

            anc_idxs = idxs[ii_frm:ii_to]

            anc = cp.asarray(self.keypoints[anc_idxs])

            # Euclidean distance between points
            M = ((kp[None, :, None, :2] - anc[:, None, :, :2]) ** 2).sum(-1) ** .5

            # Dot product between orientational unit vector
            D = (kp[None, :, None, 2:4] * anc[:, None, :, 2:4]).sum(-1)

            sim = (1 - M) * D * kp[None, :, None, 4]
            similarity[ii_frm:ii_to] = cp.asnumpy(cp.mean(cp.max(sim, axis=2), axis=1))

        ranked_idxs = idxs[np.argsort(similarity)[::-1]]
        ranked_ids = self.idx2id[ranked_idxs]

        return ranked_ids, np.sort(similarity)[::-1]

    def __init_orb_clouds(self):

        imgfiles = sorted(os.listdir(f"{self.ROOT}/images"))

        clouds = np.zeros((len(imgfiles), self.N_FEATURES, 5))
        crd_id = np.zeros((len(imgfiles),), dtype=np.long)
        # n_kp = np.zeros((len(imgfiles),), dtype=np.int)

        for ii, imgfile in enumerate(tqdm(imgfiles)):
            imgpath = f"{self.ROOT}/images/{imgfile}"

            card = Card(path=imgpath)
            img = cv.cvtColor(card.artwork, cv.COLOR_BGR2GRAY)

            h, w = img.shape[:2]

            for jj, kp in enumerate(self.orb_algorithm.detect(img, None)):
                clouds[ii, jj] = self.kp2vec(kp, w, h)
            crd_id[ii] = int(imgfile.split('_')[0])

        np.savez(f"{self.ROOT}/orb.npz", orb_clouds=clouds, card_id=crd_id)

        return clouds, crd_id

    def kp2vec(self, kp, w, h):
        return np.array([kp.pt[0] / w, kp.pt[1] / h,
                         cos(radians(kp.angle)), sin(radians(kp.angle)), 1])

# orb = cv.ORB_create()
#
# root = "res/card_database/images"
# for imgpth in os.listdir(root):
#     imgpth = f"{root}/{imgpth}"
#
#     img = cv.imread(imgpth, 0)
#
#     img = img[108:458, 26:395]
#
#     kp = orb.detect(img, None)
#     kp, des = orb.compute(img, kp)
#
#     img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
#
#     show(img2)
