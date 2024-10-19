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

analog_model = build_model(args)
analog_model = load_model(analog_model,args,analog=False)
analog_model = convert_model(analog_model,args,rpu='baseline')

analog_model_DC = build_model(args)
analog_model_DC = load_model(analog_model_DC,args,analog=False)
analog_model_DC = convert_model(analog_model_DC,args,rpu='pcm')

analog_model_HWA_DC = build_model(args)
analog_model_HWA_DC = convert_model(analog_model_HWA_DC,args,rpu='pcm')
analog_model_HWA_DC = load_model(analog_model_HWA_DC,args,analog=True)

analog_model_HWA = copy.deepcopy(analog_model_HWA_DC)
analog_model_HWA.replace_rpu_config(baseline_rpu)

train_loader, test_loader = load_data(args)

t_inferences = [1., 60., 3600., 86400., 2592000., 31104000.]
n_rep=5

digital_model.eval()
analog_model.eval()
analog_model_DC.eval()
analog_model_HWA.eval()
analog_model_HWA_DC.eval()


results_digital_model = np.zeros((len(t_inferences),n_rep))
results_analog_model = np.zeros((len(t_inferences),n_rep))
results_analog_model_DC = np.zeros((len(t_inferences),n_rep))
results_analog_model_HWA = np.zeros((len(t_inferences),n_rep))
results_analog_model_HWA_DC = np.zeros((len(t_inferences),n_rep))


baseline_acc = testnetwork(digital_model, args, test_loader)

for i,t in enumerate(t_inferences):
    print(f"t={t}")
    for id_test in range(n_rep):
        analog_model.drift_analog_weights(t)
        analog_model_DC.drift_analog_weights(t)
        analog_model_HWA.drift_analog_weights(t)
        analog_model_HWA_DC.drift_analog_weights(t)
        results_digital_model[i,id_test]=testnetwork(digital_model, args, test_loader)
        results_analog_model[i,id_test]=testnetwork(analog_model, args, test_loader)
        results_analog_model_DC[i,id_test]=testnetwork(analog_model_DC, args, test_loader)
        results_analog_model_HWA[i,id_test]=testnetwork(analog_model_HWA, args, test_loader)
        results_analog_model_HWA_DC[i,id_test]=testnetwork(analog_model_HWA_DC, args, test_loader)

column_base = f'{args.paradigm}_{args.net}{"_analog"}'
folder_path = f'./results/'
file_name = f'result_drift_acc_t_{args.n_timesteps}.csv'
path = folder_path + file_name
append_column(path,f'mean_{args.paradigm}_{args.net}_digital',np.mean(results_digital_model,axis=1))
append_column(path,f'mean_{args.paradigm}_{args.net}',np.mean(results_analog_model,axis=1))
append_column(path,f'mean_{args.paradigm}_{args.net}_DC',np.mean(results_analog_model_DC,axis=1))
append_column(path,f'mean_{args.paradigm}_{args.net}_HWA',np.mean(results_analog_model_HWA,axis=1))
append_column(path,f'mean_{args.paradigm}_{args.net}_HWA_DC',np.mean(results_analog_model_HWA_DC,axis=1))

append_column(path,f'std_{args.paradigm}_{args.net}_digital',np.std(results_digital_model,axis=1))
append_column(path,f'std_{args.paradigm}_{args.net}',np.std(results_analog_model,axis=1))
append_column(path,f'std_{args.paradigm}_{args.net}_DC',np.std(results_analog_model_DC,axis=1))
append_column(path,f'std_{args.paradigm}_{args.net}_HWA',np.std(results_analog_model_HWA,axis=1))
append_column(path,f'std_{args.paradigm}_{args.net}_HWA_DC',np.std(results_analog_model_HWA_DC,axis=1))


