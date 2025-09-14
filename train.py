import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from models.snn.utils import spike_coding, reset_net
from utils.utils import seed_everything

import time

from models.build_model import build_model, load_model
from prepare_data import load_data

from utils import get_logger, load_config, dict_to_namespace, save_config

def main(args):
    seed_everything(0)

    logger = get_logger(args.log+'/'+ args.dataset.name + '.log')
    logger.info('Start training!')

    save_names = args.log+'/'+args.dataset.name + '_' + args.paradigm + '_' + args.net + '.pth'

    model = build_model(args)

    train_loader, test_loader = load_data(args)

    if args.analog:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.regroup_param_groups(model)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    enable_scaler = not args.analog # AIHWKIT does not support mix-precison training
    scaler = torch.amp.GradScaler('cuda' if is_cuda else 'cpu', enabled=enable_scaler)
    loss_func = nn.CrossEntropyLoss()
    model = model.to(device)

    if args.results.checkpoint_path != 'none':
        start_epoch, best_val_acc = load_checkpoint(args.results.checkpoint_path, model, optimizer, scheduler)
        logger.info('checkpoint loaded: {}. resume at epoch {}, validation accuracy {:.3f}'.format(args.results.checkpoint_path, start_epoch+1, best_val_acc))
    else:
        _, _ = load_checkpoint(args.results.checkpoint_path, model)
        start_epoch, best_val_acc = 0, 0.
        logger.info('start training from scratch')

    best_val_loss = 1e3

    for id_epoch in range(start_epoch,start_epoch+args.epochs):
        time_start = time.time()
        running_loss = 0
        total_train_batches = len(train_loader)
        total_test_batches = len(test_loader)
        for batch_id, (inputs, targets) in enumerate(train_loader):
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
        for  batch_id, (inputs, targets) in enumerate(test_loader):
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
        time_epoch = time.time() - time_start

        
        is_best = val_acc > best_val_acc
        best_val_loss = min(val_loss,best_val_loss)
        best_val_acc = max(val_acc,best_val_acc)
        logger.info('Epoch:[{}/{}]\t Best Acc={:.3f}\t, Best Loss={:.6f}, Time={:.3f}s'.format(id_epoch+1 , args.epochs, best_val_acc,best_val_loss,time_epoch))

        if is_best:
            save_checkpoint(save_names, model, optimizer, scheduler, id_epoch, best_val_acc)



def train_step(model, inputs, targets, optimizer, scaler, loss_func):
    model.train()
    with torch.autocast(device_type=device.type):
        outputs = model(inputs) # snn outputs [T,bs,num_cls]
        if len(outputs.shape) == 3:
            running_loss = 0
            for t in range(outputs.shape[0]):
                running_loss += loss_func(outputs[t], targets)
            loss = running_loss/outputs.shape[0]
        else: 
            loss = loss_func(outputs.reshape(-1, outputs.size(-1)), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.detach().item()

def val_step(model, inputs, targets, loss_func):
    model.eval()
    outputs = model(inputs)
    if len(outputs.shape) == 3:
        running_loss = 0
        for t in range(outputs.shape[0]):
            running_loss += loss_func(outputs[t], targets)
        loss = running_loss/outputs.shape[0]
        _, predicted = torch.max(torch.mean(outputs,dim=0).data, 1)
    else:
        loss = loss_func(outputs.reshape(-1,outputs.size(-1)), targets)
        _, predicted = torch.max(outputs.data, 1)
    total = predicted.numel()
    correct = (predicted == targets).sum().item()
    return loss.detach().item(),total,correct

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    ckpt = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", None)
    return epoch, best_acc
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        is_cuda = True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        is_cuda = False
    else:
        device = torch.device("cpu")
        is_cuda = False
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    args = dict_to_namespace(config)

    if config["results"]["save_dir"] == "none":
        args.results.save_dir = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # else:
    #     args.results.save_dir = args.dataset.name+'-'+args.training.model+'-'+args.neuron.name+'T'+str(args.neuron.T)+'G'+str(args.neuron.num_bit)+'-'+'Loss-'+args.loss.name+'-seed'+str(args.training.seed)
    args.log = args.results.save_dir_base+'/'+args.results.save_dir
    os.makedirs(args.log, exist_ok=True)
    save_config(config, args.log)
    main(args)