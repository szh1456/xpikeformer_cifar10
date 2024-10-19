import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib import font_manager
font_path = '/scratch/users/k2363089/fonts/LiberationSerif-Regular.ttf'
font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

from utils.plotting import plot_group

import numpy as np

from parameters import parameter_reading
args = parameter_reading()

folder_path = f'./results/'
data_file_name = f'result_timesteps_acc.csv'
data_file_path = folder_path + data_file_name
fig_path = f'plots/timesteps_acc_{args.paradigm}_{args.net}.png'

timesteps = [i+1 for i in range(12)]


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

group1 = (
    (f'ann_{args.net}_digital',f'Baseline ANN (Digital)','--','black'),
    )

group2 = (
    (f'snn_{args.net}_digital',f'SNN (Digital)','--',colors[4]),
    (f'snn_{args.net}_analog','Xpikeformer','-',colors[3]),
    )

fig, ax = plt.subplots(figsize=(5,6))
plot_group(group1, timesteps, ax, data_file_path,interval=False)
plot_group(group2, timesteps, ax, data_file_path,interval=True)

# ax.set_title(f'Sample Detection Bit Error Rate\n(# Pre-training tasks: {2 ** args.start_task})')
ax.set_xlabel('Spike Encoding length')
ax.set_ylabel('Accuracy')
ax.set_xticks(timesteps)
ax.set_xlim(4,12)
ax.set_ylim(0.65,0.875)
ax.legend(prop=font_prop)
ax.grid(True,which="both",ls="--",linewidth=0.5)
fig.tight_layout(pad=.3)
fig.savefig(fig_path)
