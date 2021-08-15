import torch

from src.ml.trainer import TripletTrainer
from src.utils.tools import pyout

torch.set_printoptions(precision=3, threshold=1, edgeitems=4, sci_mode=False)

device = torch.device("cuda:0")
# trainer = TripletTrainer(device=device, state_dict_path="res/models/cardmodel_04.pth")
trainer = TripletTrainer(device=device, state_dict_path=None)
trainer.train(100)
pyout(*trainer.eval())
