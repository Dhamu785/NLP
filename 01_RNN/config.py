import torch.nn as nn
import torch as t
class CONFIG:
    epochs = 50
    batch_size = 32
    lr = 0.15
    report_every = 5
    loss = nn.NLLLoss()
    optimizer = t.optim.SGD