import os
import torch

import torch.nn as nn
import torch.optim as optim

from parameters import parameter_reading
# from aihwkit.optim import AnalogAdam

from models.snn.utils import spike_coding, reset_net
from utils.utils import seed_everything

import time



if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

args = parameter_reading()

def train_step(model, inputs, targets, optimizer, scaler, loss_func):
    model.train()
    with torch.autocast(device_type=device.type):
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.detach().item()

def val_step(model, inputs, targets, loss_func):
    model.eval()
    outputs = model(inputs)
    loss = loss_func(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)
    total = predicted.numel()
    correct = (predicted == targets).sum().item()
    return loss.detach().item(),total,correct

def save_model(model):
    folder_path = f'./model_parameters{"/analog" if args.analog else ""}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 
    if args.paradigm != 'ann':
        path = f"{folder_path}/{args.paradigm}_{args.net}_{args.encoding}.pth"
    else:
        path = f"{folder_path}/{args.paradigm}_{args.net}.pth"
    torch.save(model.state_dict(),path)
    print(f'\r Model saved at {path}.')


def trainnetwork(model, args, trainloader, testloader):
    seed_everything(0)
    enable_scaler = not args.analog # AIHWKIT does not support mix-precison training
    scaler = torch.cpu.amp.GradScaler(enabled=enable_scaler)
    if args.analog:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.regroup_param_groups(model)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_func = nn.CrossEntropyLoss()
    model = model.to(device)

    best_val_loss = 1e3
    best_val_acc = 0.

    for id_epoch in range(args.epochs):
        time_start = time.time()
        running_loss = 0
        total_train_batches = len(trainloader)
        total_test_batches = len(testloader)
        for batch_id, (inputs, targets) in enumerate(trainloader):
            if args.paradigm != 'ann':
                is_stochastic_encoding = True if args.encoding == 'b' else False
                inputs = spike_coding(inputs, args.n_timesteps, is_stochastic_encoding)
                reset_net(model)
            loss = train_step(
                model,
                inputs=inputs.to(device),
                targets=targets.to(device),
                optimizer=optimizer,
                scaler=scaler,
                loss_func=loss_func
                )
            running_loss = running_loss + loss
            percent_complete = ((batch_id + 1) / total_train_batches) * 100
            print(f'\r Epoch: {id_epoch+1}\t Training Progress: {percent_complete:.2f}%', end='')

        correct = 0
        total = 0
        for  batch_id, (inputs, targets) in enumerate(testloader):
            if args.paradigm != 'ann':
                inputs = spike_coding(inputs, args.n_timesteps, is_sto=False)
                reset_net(model)
            val_loss, total_batch, correct_batch = val_step(model,
                                        inputs=inputs.to(device),
                                        targets=targets.to(device),
                                        loss_func=loss_func
                                        )
            correct += correct_batch
            total += total_batch
            percent_complete = ((batch_id + 1) / total_test_batches) * 100
            print(f'\r Epoch: {id_epoch+1}\t Evaulating Progress: {percent_complete:.2f}%', end='')
        val_acc = correct / total
        
        time_elapsed = time.time() - time_start
        if val_acc > best_val_acc:
            save_model(model)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch_id = id_epoch+1
            print(f'\r Epoch : {id_epoch+1} -- New Best Model with Validation Loss: {val_loss} and Accuracy: {val_acc}. Time used: {time_elapsed:.2f} seconds')
        else:
            print(f'\r Epoch : {id_epoch+1} -- Best model at Epoch {best_epoch_id} with Validation Loss: {best_val_loss} and Accuracy: {best_val_acc}. Time used: {time_elapsed:.2f} seconds')
