from parameters import parameter_reading
from models.build_model import build_model, load_model, convert_model
from prepare_data import load_data
import os
import time
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parameter_reading()
time_start = time.time()

model = build_model(args)
if args.analog:
    model = convert_model(model,args,rpu='pcm')
    model = load_model(model,args,analog=True)
else:
    model = load_model(model,args,analog=False)
train_loader, test_loader = load_data(args)
if args.mode == 'train':
    print(f"Start training")
    from train import trainnetwork
    trainnetwork(model, args, train_loader, test_loader)
    print("Training is done.")
elif args.mode == 'test':
    print(f"Start testing")
    from test import testnetwork
    testnetwork(model, args, test_loader)
    print("Testing is done.")
del model
torch.cuda.empty_cache()
time_elapsed = time.time() - time_start
print(f'\nTotal time elapsed: {time_elapsed:.2f} seconds')
