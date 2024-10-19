import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
matplotlib.rcParams['font.family'] = 'Calibri'

from utils.plotting import plot_group

import numpy as np

from parameters import parameter_reading
args = parameter_reading()

folder_path = f'./results/'
data_file_name = f'result_drift_acc_t_{args.n_timesteps}.csv'
data_file_path = folder_path + data_file_name
fig_path = f'plots/{args.paradigm}_{args.net}.png'

t_inferences = [1., 60., 3600., 86400., 2592000., 31104000.]


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
labels = ['1s', '1min', '1hr', '1day', '1mon', '1yr']

# group1 = (
#     (f'{args.paradigm}_{args.net}_digital',f'Digital ANN Baseline','--','black'),
#     )

results = (
    (f'{args.paradigm}_{args.net}','CT+NC','-',colors[0]),
    (f'{args.paradigm}_{args.net}_DC','CT+GDC','-',colors[1]),
    (f'{args.paradigm}_{args.net}_HWA','HWAT+NC','-',colors[2]),
    (f'{args.paradigm}_{args.net}_HWA_DC','HWAT+GDC','-',colors[3]),
    )



fig, ax = plt.subplots(figsize=(5,6))
# plot_group(group1, t_inferences, ax, data_file_path,interval=False)
plot_group(results, t_inferences, ax, data_file_path,interval=True)

# ax.set_title(f'Sample Detection Bit Error Rate\n(# Pre-training tasks: {2 ** args.start_task})')
ax.set(xlabel='Time after Programming', ylabel='Accuracy')
ax.set_xscale('log')
ax.set_xticks(t_inferences)
ax.set_xticklabels(labels)
ax.set_ylim(0.65,0.875)
ax.legend()
ax.grid(True,which="both",ls="--",linewidth=0.5)
fig.tight_layout(pad=.3)
fig.savefig(fig_path)
