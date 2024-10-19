import pandas as pd

def plot_group(tuple_to_plot, x_values, ax, data_file_path, interval=False):
    df = pd.read_csv(data_file_path)
    for name, label, style, color in tuple_to_plot:
        if 'mean_'+name in df.columns:
            if interval:
                ax.errorbar(x_values, df['mean_'+name], yerr=df['std_'+name], fmt=style, label=label, elinewidth=2, capsize=4, color=color,linewidth=2) # , color='blue', ecolor='lightblue'
            else:
                ax.plot(x_values, df['mean_'+name], style, label=label,color=color,linewidth=2)
        else:
            print(f"Column '{name}' does not exist in the DataFrame.")