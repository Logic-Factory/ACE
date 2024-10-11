import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import matplotlib.pyplot as plt

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf',  
            '#7f7f7f', '#e377c2', '#8c564b', '#bcbd22']
ecolors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#9edae5', 
            '#c7c7c7', '#f7b6d2', '#c49c94', '#dbdb8d']
            
tex_fonts = {
    "font.family": "serif",
    'font.variant': "small-caps",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "lines.linewidth":2.,
    "lines.markersize":4,
    
    "axes.grid": True, 
    "grid.color": ".9", 
    "grid.linestyle": "--",

    "axes.linewidth":1.5, 
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'savefig.facecolor': 'white'
}        

# plt.style.use('fivethirtyeight')
plt.rcParams.update(tex_fonts)
plt.rcParams['figure.dpi'] = 600

# style={ 0:'-', 1: '-', 2:'-.', 3:'--',  4:':', 5:'-.', 6:'--', 7: ':'}
style={ 0:'--', 1: '-', 3:':', 4:'dashdot',  2:'-', -1:'-'}
color={ 0:'C2', 1: 'C3', 3: 'C0', 4:'C1', 5:'C4', 2:'C5'}

linestyle_tuple = ['-', ':', '-.', '--', '-.','-']
markers = [
    "*", "+", ".", "x", "d", "p", "|",
    "o", "s", "^", "<", "v", ">",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "c", "m", "y", "k", "r", "g", "b",
    "-", "--", ":", ".-.",
    "none",
]

def plot_curve(lists, labels, title, x_label, y_label, save_path):
    plt.clf()
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6, 4))
    axes.grid(False)
    
    x = range(len(lists[0]))
    for i, y in enumerate(lists):
        axes.plot(x, y, color=colors[i % len(colors)], linestyle=style[i % len(style)], marker = markers[i % len(markers)], markevery=5, linewidth=1.5, alpha=0.8)
    
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    if len(labels) > 1:
        axes.legend(loc='upper right', fancybox = True, shadow = False, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_2d_dots(x_list, y_list, title, x_label, y_label, save_path):
    plt.clf()
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6, 4))
    axes.grid(False)
       
    plt.scatter(x_list, y_list, marker=markers[0], color=colors[0])

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_multi_2d_dots(x_lists, y_lists, labels, title, x_label, y_label, save_path):
    plt.clf()
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6, 4))
    axes.grid(False)
    
    for i in range(len(x_lists)):
        plt.scatter(x_lists[i], y_lists[i], marker=markers[i % len(markers)], color=colors[i % len(colors)], label=labels[i])

    if len(labels) > 1:
        axes.legend(loc='center left', fancybox = True, shadow = False, bbox_to_anchor=(1, 0.5))

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def plot_3d_dots(x_list, y_list, z_list, title, x_label, y_label, z_label, save_path):
    plt.clf()
    
    fig = plt.figure(figsize=(6, 4))
    axes = fig.add_subplot(111, projection='3d')
    
    scatter = axes.scatter(x_list, y_list, z_list, c=z_list, cmap='viridis', s=50, alpha=0.8)

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_zlabel(z_label)

    axes.view_init(elev=30, azim=45)

    color_bar = plt.colorbar(scatter, ax=axes, shrink=0.5, aspect=10)
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
