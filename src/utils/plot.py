import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import scipy
from scipy.spatial import ConvexHull

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
        label = labels[i] if i < len(labels) else None
        axes.plot(x, y, color=colors[i % len(colors)], linestyle=style[i % len(style)], label=label, marker = markers[i % len(markers)], markevery=5, linewidth=1.5, alpha=0.8)
    
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    if len(labels) > 1:
        # axes.legend(loc='upper right', fancybox = True, shadow = False, bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        axes.legend(fancybox = True, shadow = False, borderaxespad=0.)
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_dual_curve(list0, list1, title, x_label, y0_label, y1_label, save_path):
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Create the first axis for loss
    ax1.plot(list0, color='tab:red', linestyle='-', marker='x', markersize=5, label=y0_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y0_label, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second axis for accuracy, sharing the same x-axis
    ax2 = ax1.twinx()
    line3, = ax2.plot(list1, color='tab:blue', linestyle='-', marker='x', markersize=5, label=y1_label)
    ax2.set_ylabel(y1_label, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title and legend
    plt.title(title)

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

def plot_2d_heatmap(x_list, y_list, title, x_label, y_label, save_path, fontsize = 10, cmap_coplor='Greens'):
    plt.clf()
    fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6, 4))
    axes.grid(False)
       
    heatmap, xedges, yedges = np.histogram2d(x_list, y_list, bins=30)
    im = axes.imshow(heatmap.T, origin='lower', cmap=cmap_coplor, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))  # 设置科学计数法的阈值
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes)
    cbar.set_label('Count')
    
    axes.set_xlabel(x_label, fontsize=fontsize)
    axes.set_ylabel(y_label, fontsize=fontsize)
    
    if title != '':
        axes.set_title(title, fontsize=fontsize)
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_multi_2d_dots(x_lists, y_lists, labels, title, x_label, y_label, save_path, fontsize = 10, isLegend = True):
    plt.clf()
    fig, axes = plt.subplots(figsize=(10, 8))
    
    assert len(x_lists) == len(y_lists) == len(labels), "Length of x_lists, y_lists, and labels must be the same."
    
    for x, y, label, color in zip(x_lists, y_lists, labels, colors):
        data = np.column_stack((x, y))
        axes.scatter(x, y, label=label, color=color, alpha=0.5)
        
        if len(x) > 2 and np.unique(data, axis=0).shape[0] > 1:
            try:
                hull = ConvexHull(data)
                # 绘制凸包的边
                for simplex in hull.simplices:
                    axes.plot(data[simplex, 0], data[simplex, 1], color=color)
            except scipy.spatial.qhull.QhullError:
                print("Cannot create a convex hull for the data with all points coincident.")

    if isLegend:
        axes.legend(loc='center left', fancybox = True, shadow = False, bbox_to_anchor=(1, 0.5), fontsize=fontsize)

    axes.set_xlabel(x_label, fontsize=fontsize)
    axes.set_ylabel(y_label, fontsize=fontsize)
    
    if title != '':
        axes.set_title(title, fontsize=fontsize)
    
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

def plot_3d_heatmap(x_list, y_list, z_list, title, x_label, y_label, z_label, save_path):
    plt.clf()
    
    fig = plt.figure(figsize=(6, 4))
    axes = fig.add_subplot(111, projection='3d')

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)
    
    # 计算综合评分
    scores = 1 / (0.33 * x_list + 0.34 * y_list + 0.33 * z_list)

    # 归一化分数
    norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))    
    scores_count = np.bincount(np.digitize(scores, np.histogram_bin_edges(scores, bins=10)))
    freq = scores_count[np.digitize(scores, np.histogram_bin_edges(scores, bins=10)) - 1]  # 频率为对应分数的点数量

    # 使用颜色和点的大小进行可视化，大小和透明度与评分频率成正比
    scatter = axes.scatter(x_list, y_list, z_list, c=norm_scores, cmap='RdYlGn', s=freq * 20, alpha= np.clip(0.5 + 0.5 * norm_scores, 0, 1))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    axes.zaxis.set_major_formatter(formatter)

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_zlabel(z_label)

    plt.title(title)
    axes.view_init(elev=30, azim=45)

    color_bar = plt.colorbar(scatter, ax=axes, shrink=0.5, aspect=10)
    color_bar.set_label('Performance Score (Higher is Better)')
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def plot_matrix(matrix, x_labels, y_labels, save_path, cmap_color:str = 'coolwarm'):
    colors = ["#ffffff", "#ff6666", "#ff0000"]  # white -> red
    # colors = ["#ffffff", "#806666", "#800000"]  # white -> Maroon
    cmap_name = "linear_red"
    linear_red = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    if cmap_color == 'coolwarm':
        plt.imshow(matrix, cmap='coolwarm', interpolation='none')
    else:
        plt.imshow(matrix, cmap=linear_red, interpolation='none')
    plt.colorbar(shrink=0.8)
    if x_labels:
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, ha="right", rotation_mode="anchor", rotation=45, fontsize=6)
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, rotation=45, fontsize=6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()