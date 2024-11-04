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

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from dataset import ClassificationDataset
from net import ClassificationNet

from src.utils.feature import padding_feature_to
from src.utils.plot import plot_curve

from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, root_openlsd:str, processed_dir:str, designs:List[str], logic:str, recipes:int, dim_input:int, dim_hidden:int, epoch_size:int, batch_size:int, workspace:str):
        self.root_openlsd = root_openlsd
        self.processed_dir = processed_dir
        self.designs = designs
        self.logic = logic
        self.recipes = recipes
        
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.epoch = epoch_size
        self.batch_size = batch_size
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, logic, self.current_time)

        
        # load the dataset first
        self.dataset = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic=self.logic, recipes=self.recipes)
        self.model = ClassificationNet(dim_input, dim_hidden, self.dataset.num_classes()).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        print("Configuration of Circuit Classification:")
        print(f"root_openlsd: {self.root_openlsd}")
        print(f"processed_dir: {self.processed_dir}")
        print(f"designs: {self.designs}")
        print(f"logics: {self.logic}")
        print(f"recipes: {self.recipes}")
        print(f"dim_input: {self.dim_input}")
        print(f"dim_hidden: {self.dim_hidden}")
        print(f"epoch: {self.epoch}")
        print(f"batch_size: {self.batch_size}")
        print(f"workspace: {self.workspace}")
        print(f"device: {device}")
    
    
    def preproccess_graph(self):
        for graph in self.dataset:
            graph = padding_feature_to(graph, self.dim_input)
    
    def run(self):
        self.preproccess_graph()
        
        dataset_train, dataset_test = self.dataset.split_train_test(0.8)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)
        
        torch.manual_seed(12345)
        total_loss_train, total_acc_train = [], []
        total_loss_test, total_acc_test = [], []
        
        os.makedirs(self.workspace, exist_ok=True)
        
        print("Start training:")
        for epoch in range(self.epoch):
            loss_train, acc_train = self.train(dataloader_train)
            loss_test, acc_test = self.eval(dataloader_test)
            total_loss_train.append(loss_train)
            total_acc_train.append(acc_train)
            total_loss_test.append(loss_test)
            total_acc_test.append(acc_test)
            
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.4f} ({loss_test:.4f}), Acc: {acc_train:.4f} ({acc_test:.4f})')
            plot_curve(lists= [total_loss_train], labels=[], title="train loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, "loss_train.pdf"))
            plot_curve(lists= [total_acc_train], labels=[], title="train acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_train.pdf"))
            plot_curve(lists= [total_loss_test], labels=[], title="test loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace,"loss_test.pdf"))
            plot_curve(lists= [total_acc_test], labels=[], title="test acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_test.pdf"))
            
        self.tsne_analysis(dataloader_train, os.path.join(self.workspace, "tsne_train.pdf"))
        self.tsne_analysis(dataloader_test, os.path.join(self.workspace, "tsne_test.pdf"))

    def train(self, dataloader:DataLoader):
        self.model.train()
        total_acc = 0
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_acc += pred.eq(data.y).sum().item()
        return total_loss / len(dataloader.dataset), total_acc / len(dataloader.dataset)

    def eval(self, dataloader:DataLoader):
        self.model.eval()
        total_acc = 0
        total_loss = 0
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                out = self.model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                total_acc += pred.eq(data.y).sum().item()
        return total_loss / len(dataloader.dataset), total_acc / len(dataloader.dataset)
    
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
    parser.add_argument('--root_openlsd', type=str, required=True, help='the path of the datapath')
    parser.add_argument('--recipe_size', type=int, default=50, help='the extracted recipe size for each design')
    parser.add_argument('--processed_dir', type=str, help='the dimenssion of the feature size for each node')
    parser.add_argument('--logic', type=str, default="abc", help='the logic type of the selected dataset')
    parser.add_argument('--dim_input', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=128, help='the dimension of the hidden layer')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_designs = [
        "ctrl",
        # "steppermotordrive",
        "router",
        # "int2float",
        "ss_pcm",
        # "usb_phy",
        "sasc",
        ]
    
    trainer = Trainer(root_openlsd=args.root_openlsd,
                      processed_dir=args.processed_dir,
                      designs=curr_designs,
                      logic=args.logic,
                      recipes=args.recipe_size,
                      dim_input=args.dim_input,
                      dim_hidden=args.dim_hidden,
                      epoch_size=args.epoch_size,
                      batch_size=args.batch_size,
                      workspace=args.workspace)
    trainer.run()