import torch

import torch.nn as nn
import torch.optim as optim

from models.snn.utils import spike_coding, reset
from utils.utils import seed_everything

import time

import spikingjelly.activation_based.monitor as monitor
from train import val_step

import numpy as np


def testnetwork(model, args, testloader):
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    model = model.to(device)
    best_val_acc = .0
    correct = 0
    total = 0
    for id, (inputs, targets) in enumerate(testloader):
        if args.paradigm != 'ann':
            inputs = spike_coding(inputs, args.n_timesteps, is_sto=(False if args.encoding == 'u' else True))
            reset(model)
        val_loss, total_batch, correct_batch = val_step(model,
                                    inputs=inputs.to(device),
                                    targets=targets.to(device),
                                    loss_func=loss_func
                                    )
        correct += correct_batch
        total += total_batch
    val_acc = correct / total
    print(f'Testing Accuracy: {val_acc*100:.2f}%.')
    return val_acc
