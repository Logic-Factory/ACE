import os.path
import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import torch
import torch.nn as  nn
import torch.nn.functional as F
from torch.optim.adam import Adam

# from net import ClassificationNet
from net import GraphSAGE_NET,get_model
from dataset import Probability_prediction
from src.utils.plot import plot_curve
from dataset import PP_Data #necessary
from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import sys



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, args):
        self.workspace = args.workspace
        self.model_save = args.model_save
        self.logic = args.logic
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.model_save, exist_ok=True)
        os.makedirs(os.path.join(self.workspace, self.logic), exist_ok=True)
        os.makedirs(os.path.join(self.model_save, self.logic), exist_ok=True)
        self.epoch = args.epoch_size
        self.batch_size = args.batch_size
        # self.dataset = Probability_prediction(args.root, args.recipe_size, args.white_design_list, args.logic, args.dim_input)
        # self.dataset = Probability_prediction(args.root, args.recipe_size, curr_designs=args.white_design_list, processed_dir=args.target)
        self.dataset = Probability_prediction(args.root,args.target, args.white_design_list,'abc',args.recipe_size )
        args.device = device
        self.model = get_model(args)
        self.model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        # self.loss_func = nn.MSELoss()
        self.loss_func = nn.L1Loss(reduction='mean')
        self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
    def run(self):
        # self.dataset.print_data_list()
        dataset_train, dataset_test = self.dataset.split_train_test(0.8)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)
        
        torch.manual_seed(12345)
        total_loss_train, total_acc_train = [], []
        total_loss_test, total_acc_test = [], []
        stime = time.time()
        for epoch in range(self.epoch):
            loss_train = self.train(dataloader_train)
            st = time.time()
            loss_test = self.eval(dataloader_test)
            et = time.time()
            print("test time: ",str(et - st))
            total_loss_train.append(loss_train)
            total_loss_test.append(loss_test)
            print("loss_train",loss_train)
            print("loss_train",loss_test)
            torch.save(self.model,os.path.join(self.model_save,self.logic,self.current_time+str(epoch) + "_model.pt"))
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.4f} ({loss_test:.4f})')
            plot_curve(lists= [total_loss_train], labels=[], title="train loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, self.logic, self.current_time + "_loss_train.pdf"))
            # plot_curve(lists= [total_acc_train], labels=[], title="train acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, self.logic, self.current_time + "_acc_train.pdf"))
            plot_curve(lists= [total_loss_test], labels=[], title="test loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, self.logic, self.current_time + "_loss_test.pdf"))
 
            # plot_curve(lists= [total_acc_test], labels=[], title="test acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, self.logic, self.current_time + "_acc_test.pdf"))
            
        # self.tsne_analysis(dataloader_train, os.path.join(self.workspace, self.logic, self.current_time + "_tsne_train.pdf"))
        # self.tsne_analysis(dataloader_test, os.path.join(self.workspace, self.logic, self.current_time + "_tsne_test.pdf"))

    def train(self, dataloader:DataLoader):
        self.model.train()
        total_acc = 0
        total_loss = 0
        print('len(dataloader)',len(dataloader))
        for data in tqdm(dataloader):
            # print(data)
            data = data.to(device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.loss_func(out, data.label.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader.dataset)

    def eval(self, dataloader:DataLoader):
        self.model.eval()
        total_acc = 0
        total_loss = 0
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                out = self.model(data)
                loss = self.loss_func(out, data.label.unsqueeze(1))
                total_loss += loss.item()
        return total_loss / len(dataloader.dataset)
    
    def tsne_analysis(self, dataloader:DataLoader, output_path):
        # extract features
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                out = self.model(data.x, data.edge_index, data.batch)
                features.append(out.cpu().numpy())
                labels.append(data.y.cpu().numpy())
        features = np.vstack(features)
        labels = np.hstack(labels)

        # Adjusting perplexity based on the sample size
        n_samples = features.shape[0]
        perplexity = min(30, max(5, int(n_samples / 3)))  # Perplexity between 5 and 30, adjusted to dataset size

        # extract the mainly feature by PCA
        pca = PCA(n_components=min(features.shape[1], 50))
        features_pca = pca.fit_transform(features)

        # tsne manipulation computation
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_res = tsne.fit_transform(features_pca)
        
        # plot the distribution of the features
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c=labels, cmap='tab20')
        legend = ax.legend(*scatter.legend_elements(), title="Classes", loc = 'upper right', fancybox=True, shadow=False)
        ax.add_artist(legend)
        plt.savefig(output_path)
        plt.close()
    
if __name__ == '__main__':
    # config the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='the path of the datapath')
    parser.add_argument('--target', type=str, default='./data_pp', help='the path of the target datapath')
    parser.add_argument('--workspace', type=str, default="./workspace", help='the path of the workspace to store the results')
    parser.add_argument('--model_save', type=str, default="./pt", help='the path of the workspace to store the results')
    parser.add_argument('--logic', type=str, default="abc", help='the logic type of the selected dataset')
    parser.add_argument('--recipe_size', type=int, default=100, help='the extracted recipe size for each design')
    parser.add_argument('--dim_input', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=128, help='the dimension of the hidden layer')
    parser.add_argument('--epoch_size', type=int, default=60, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size of the dataloader')
    parser.add_argument('--num_rounds', type=int, default=1, help='num_rounds')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--dim_mlp', type=int, default=64, help='dim_mlp')
    parser.add_argument('--dim_pred', type=int, default=1, help='dim_pred')

    
    args = parser.parse_args()
    args.root = '/data/project_share/openlsd_1028'
    args.batch_size = 64
    args.white_design_list = [
        "ctrl",
        "steppermotordrive",
        "router",
        "int2float",
        "ss_pcm",
        "usb_phy",
        "sasc",
        "cavlc",
        "simple_spi",
        "priority"]
    trainer = Trainer(args)
    trainer.run()