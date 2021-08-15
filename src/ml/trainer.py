import os
import time

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.cardartdataset import CardArtTrainSet, CardArtEvalSet
from src.ml.cardmodel import CardModel
from src.ml.loss_functions import semi_hard_triplet_loss, batchwise_triplet_loss
from src.utils.tools import poem, pyout, contains_nan


class TripletTrainer:
    def __init__(self, state_dict_path=None, device=None):
        small = torch.cuda.device_count() <= 1
        batch_size = 16 if small else 64
        num_workers = 0 if small else 8

        self.trainset = CardArtTrainSet()
        self.trainloader = DataLoader(self.trainset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

        self.eva_set = CardArtEvalSet()
        self.eval_loader = DataLoader(self.trainset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

        self.model = CardModel(architecture="mobile" if small else "resnet")
        if state_dict_path is not None:
            self.model.load_state_dict(torch.load(state_dict_path))
        self.model.to(device)
        self.device = device

        self.optim = Adam(self.model.parameters(), lr=4e-3)

        with open("res/models/log.txt", "w+") as f:
            f.write("")

        self.epoch = 0

    def train(self, epochs):
        for ii in range(self.epoch, self.epoch + epochs):
            self.model.train()

            cum_loss, cum_N = 0., 0
            pbar = tqdm(range(len(self.trainloader)), total=len(self.trainloader),
                        desc=poem(f"epoch {ii}"), leave=False)
            for A, P in self.trainloader:
                self.optim.zero_grad()
                A, P = A.to(self.device), P.to(self.device)

                A_fv = self.model(A)
                P_fv = self.model(P)

                contains_nan(A_fv)
                contains_nan(P_fv)

                loss: Tensor = batchwise_triplet_loss(A_fv, P_fv)
                # loss: Tensor = semi_hard_triplet_loss(A_fv, P_fv)
                contains_nan(loss)

                loss.backward()
                self.optim.step()

                cum_loss += loss.cpu().item()
                cum_N += A.size(0)

                pbar.update(1)

            time.sleep(1)
            with open("res/models/log.txt", 'a+') as f:
                f.write(f"\n{cum_loss / cum_N:.7f}")
            pyout(*self.eval(100), round(cum_loss / cum_N, 3))

            fname = f"res/models/cardmodel_{str(ii).zfill(2)}.pth"
            if os.path.isfile(fname):
                os.remove(fname)
            torch.save(self.model.state_dict(), fname)

    def eval(self, N=None):
        self.model.eval()
        N = len(self.trainset) if N is None else min(len(self.trainset), N)
        fv = torch.zeros(N, 1000, dtype=torch.float32)
        dp = torch.zeros(N, dtype=torch.float32)
        bs = self.eval_loader.batch_size

        for ii, (A, P) in tqdm(enumerate(self.eval_loader),
                               desc=poem("evaluating..."),
                               total=(N // bs + (N % bs != 0)),
                               leave=False):
            with torch.no_grad():
                A, P = A.to(self.device), P.to(self.device)

                A_fv = self.model(A)
                P_fv = self.model(P)

                ii_from = ii * bs
                ii_to = min(ii_from + A.size(0), N)
                n = ii_to - ii_from

                if n <= 0:
                    break

                dp[ii_from:ii_to] = (((A_fv[:n] - P_fv[:n]) ** 2).mean(1) ** .5)
                fv[ii_from:ii_to] = A_fv[:n]

        rank = torch.zeros(N)

        for ii in tqdm(range(0, N, bs), desc=poem("nearest neighbour"),
                       total=(N // bs + (N % bs != 0)), leave=False):
            ii_from = ii
            ii_to = min(ii_from + bs, N)
            dmatrix = ((fv[ii_from:ii_to, None, :] - fv[None, :, :]) ** 2).mean(dim=2) ** .5

            dmatrix[torch.arange(0, ii_to - ii_from), torch.arange(ii_from, ii_to)] = float('inf')

            rank[ii_from:ii_to] = (dmatrix <= dp[ii_from:ii_to, None]).sum(dim=1)

        self.model.train()
        return round(torch.median(rank).item(), 3), round(torch.mean(rank).item(), 3)
