from parameters import parameter_reading
from models.build_model import build_model, load_model, convert_model
from prepare_data import load_data
import os
import time
import torch
from test import testnetwork
import copy
from models.get_rpu import get_rpu
from utils.file_operation import append_column
import numpy as np


from aihwkit.nn.conversion import convert_to_analog


pcm_rpu = get_rpu('pcm')
baseline_rpu = get_rpu('baseline')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parameter_reading()
time_start = time.time()

digital_model = build_model(args)
digital_model = load_model(digital_model,args,analog=False)

analog_model_HWA_DC = build_model(args)
analog_model_HWA_DC = convert_model(analog_model_HWA_DC,args,rpu='pcm')
analog_model_HWA_DC = load_model(analog_model_HWA_DC,args,analog=True)

train_loader, test_loader = load_data(args)

timesteps = [i+1 for i in range(10)]
n_rep=2

digital_model.eval()
analog_model_HWA_DC.eval()


results_digital_model = np.zeros((len(timesteps),n_rep))
results_analog_model_HWA_DC = np.zeros((len(timesteps),n_rep))



for i,t in enumerate(timesteps):
    print(f"timesteps={t}")
    args.n_timesteps = t
    baseline_acc=testnetwork(digital_model, args, test_loader)
    for id_test in range(n_rep):
        results_digital_model[i,id_test] = baseline_acc
        analog_model_HWA_DC.drift_analog_weights(86400.)
        results_analog_model_HWA_DC[i,id_test]=testnetwork(analog_model_HWA_DC, args, test_loader)


folder_path = f'./results/'
file_name = f'result_timesteps_acc.csv'
path = folder_path + file_name
append_column(path,f'mean_{args.paradigm}_{args.net}_digital',np.mean(results_digital_model,axis=1))
append_column(path,f'mean_{args.paradigm}_{args.net}_analog',np.mean(results_analog_model_HWA_DC,axis=1))

append_column(path,f'std_{args.paradigm}_{args.net}_digital',np.std(results_digital_model,axis=1))
append_column(path,f'std_{args.paradigm}_{args.net}_analog',np.std(results_analog_model_HWA_DC,axis=1))


