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

import pandas as pd

from dataset import ClassificationDataset
from net import ClassificationNet

from src.utils.feature import padding_feature_to
from src.utils.plot import plot_curve, plot_curve2

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
        self.dataset2 = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic="oig", recipes=self.recipes)
        self.dataset3 = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic="xag", recipes=self.recipes)
        self.dataset4 = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic="mig", recipes=self.recipes)
        self.dataset5 = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic="primary", recipes=self.recipes)
        self.dataset6 = ClassificationDataset(root_openlsd=self.root_openlsd, processed_dir=self.processed_dir, designs=self.designs, logic="gtg", recipes=self.recipes)
        
        self.model = ClassificationNet(dim_input, dim_hidden, self.dataset.num_classes()).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)


        os.makedirs(self.workspace, exist_ok=True)
        self.logfile = os.path.join(self.workspace, 'log.txt')
        self.datafile = os.path.join(self.workspace, 'data.csv')

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
        
        with open(self.logfile, 'w') as f:
            f.write("Configuration of Circuit Classification:\n")
            f.write(f"root_openlsd: {self.root_openlsd}\n")
            f.write(f"processed_dir: {self.processed_dir}\n")
            f.write(f"designs: {self.designs}\n")
            f.write(f"logics: {self.logic}\n")
            f.write(f"recipes: {self.recipes}\n")
            f.write(f"dim_input: {self.dim_input}\n")
            f.write(f"dim_hidden: {self.dim_hidden}\n")
            f.write(f"epoch: {self.epoch}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"workspace: {self.workspace}\n")
            f.write(f"device: {device}\n")
    
    
    def preproccess_graph(self):
        for graph in self.dataset:
            graph = padding_feature_to(graph, self.dim_input)
        for graph in self.dataset2:
            graph = padding_feature_to(graph, self.dim_input)
        for graph in self.dataset3:
            graph = padding_feature_to(graph, self.dim_input)
        for graph in self.dataset4:
            graph = padding_feature_to(graph, self.dim_input)
        for graph in self.dataset5:
            graph = padding_feature_to(graph, self.dim_input)
        for graph in self.dataset6:
            graph = padding_feature_to(graph, self.dim_input)
    
    def run(self):
        # padding the graph feature
        print(f"padding feature ing")
        self.preproccess_graph()

        # split the dataset into training and testing
        print(f"split datasets")
        dataset_train, dataset_test = self.dataset.split_train_test(designs=self.designs, train_ratio=0.8)
        dataset_test_2, _ = self.dataset2.split_train_test(designs=self.designs, train_ratio=1.0)
        dataset_test_3, _ = self.dataset3.split_train_test(designs=self.designs, train_ratio=1.0)
        dataset_test_4, _ = self.dataset4.split_train_test(designs=self.designs, train_ratio=1.0)
        dataset_test_5, _ = self.dataset5.split_train_test(designs=self.designs, train_ratio=1.0)
        dataset_test_6, _ = self.dataset6.split_train_test(designs=self.designs, train_ratio=1.0)
        
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)
        dataloader_test_2 = DataLoader(dataset_test_2, batch_size=self.batch_size, shuffle=True)
        dataloader_test_3 = DataLoader(dataset_test_3, batch_size=self.batch_size, shuffle=True)
        dataloader_test_4 = DataLoader(dataset_test_4, batch_size=self.batch_size, shuffle=True)
        dataloader_test_5 = DataLoader(dataset_test_5, batch_size=self.batch_size, shuffle=True)
        dataloader_test_6 = DataLoader(dataset_test_6, batch_size=self.batch_size, shuffle=True)
                
        
        total_loss_train, total_acc_train = [], []
        total_loss_test, total_acc_test = [], []
        
        total_loss_test_2, total_acc_test_2 = [], []
        total_loss_test_3, total_acc_test_3 = [], []
        total_loss_test_4, total_acc_test_4 = [], []
        total_loss_test_5, total_acc_test_5 = [], []
        total_loss_test_6, total_acc_test_6 = [], []
        
        print("Start training:")
        torch.manual_seed(12345)
        for epoch in range(self.epoch):
            loss_train, acc_train = self.train(dataloader_train)
            loss_test, acc_test = self.eval(dataloader_test)
            
            total_loss_train.append(loss_train)
            total_acc_train.append(acc_train)
            total_loss_test.append(loss_test)
            total_acc_test.append(acc_test)
            
            loss_test_2, acc_test_2 = self.eval(dataloader_test_2)
            loss_test_3, acc_test_3 = self.eval(dataloader_test_3)
            loss_test_4, acc_test_4 = self.eval(dataloader_test_4)
            loss_test_5, acc_test_5 = self.eval(dataloader_test_5)
            loss_test_6, acc_test_6 = self.eval(dataloader_test_6)
            
            total_loss_test_2.append(loss_test_2)
            total_acc_test_2.append(acc_test_2)
            total_loss_test_3.append(loss_test_3)
            total_acc_test_3.append(acc_test_3)
            total_loss_test_4.append(loss_test_4)
            total_acc_test_4.append(acc_test_4)
            total_loss_test_5.append(loss_test_5)
            total_acc_test_5.append(acc_test_5)
            total_loss_test_6.append(loss_test_6)
            total_acc_test_6.append(acc_test_6)
            
            log = f'Epoch: {epoch:03d}, \nLoss: {loss_train:.4f} ({loss_test:.4f}, {loss_test_2:.4f}, {loss_test_3:.4f}, {loss_test_4:.4f}, {loss_test_5:.4f}, {loss_test_6:.4f}), \nAcc: {acc_train:.4f} ({acc_test:.4f}, {acc_test_2:.4f}, {acc_test_3:.4f}, {acc_test_4:.4f}, {acc_test_5:.4f}, {acc_test_6:.4f})'
            print(log)
            with open(self.logfile, 'a') as f:
                f.write(log + '\n')
            plot_curve2(lists= [total_loss_train], labels=[], title="train loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, "loss_train.pdf"))
            plot_curve2(lists= [total_acc_train], labels=[], title="train acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_train.pdf"))
            plot_curve2(lists= [total_loss_test], labels=[], title="test loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace,"loss_test.pdf"))
            plot_curve2(lists= [total_acc_test], labels=[], title="test acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_test.pdf"))
            
        data = pd.DataFrame({
            "train_loss": total_loss_train,
            "test_loss": total_loss_test,
            "train_acc": total_acc_train,
            "test_acc": total_acc_test,
            "test_loss_oig": total_loss_test_2,
            "test_acc_oig": total_acc_test_2,
            "test_loss_xag": total_loss_test_3,
            "test_acc_xag": total_acc_test_3,
            "test_loss_mig": total_loss_test_4,
            "test_acc_mig": total_acc_test_4,
            "test_loss_primary": total_loss_test_5,
            "test_acc_primar": total_acc_test_5,
            "test_loss_gtg": total_loss_test_6,
            "test_acc_gtg": total_acc_test_6,
        })
        data.to_csv(self.datafile, index=False)
            
        self.tsne_analysis(dataloader_train, os.path.join(self.workspace, "tsne_train.pdf"))
        self.tsne_analysis(dataloader_test, os.path.join(self.workspace, "tsne_test.pdf"))
        self.tsne_analysis(dataloader_test_2, os.path.join(self.workspace, "tsne_test_2.pdf"))
        self.tsne_analysis(dataloader_test_3, os.path.join(self.workspace, "tsne_test_3.pdf"))
        self.tsne_analysis(dataloader_test_4, os.path.join(self.workspace, "tsne_test_4.pdf"))
        self.tsne_analysis(dataloader_test_5, os.path.join(self.workspace, "tsne_test_5.pdf"))
        self.tsne_analysis(dataloader_test_6, os.path.join(self.workspace, "tsne_test_6.pdf"))
    
    
        torch.save(self.model.state_dict(), os.path.join(self.workspace, "model.pth"))

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
        handles, labels = scatter.legend_elements()
        unique = list(set(labels))
        legend = ax.legend(handles, unique, title="Classes", loc = 'upper right', fancybox=True, shadow=False)
        ax.add_artist(legend)
        plt.savefig(output_path)
        plt.close()
    
if __name__ == '__main__':
    # config the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_openlsd', type=str, required=True, help='the path of the datapath')
    parser.add_argument('--processed_dir', type=str, help='the dimenssion of the feature size for each node')
    parser.add_argument('--logic', type=str, default="abc", help='the logic type of the selected dataset')
    parser.add_argument('--recipes', type=int, default=50, help='the extracted recipe size for each design')
    parser.add_argument('--dim_input', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=128, help='the dimension of the hidden layer')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()


    curr_designs = [
        'adder',
        'bar',
        'ctrl',
        'int2float',
        'priority',
        'sin',
        'ss_pcm',
        'usb_phy',
        'cavlc',
        'i2c',
        'max',
        'router',
        'sasc',
        'spi',
        'steppermotordrive'
        ]
    
    trainer = Trainer(root_openlsd=args.root_openlsd,
                      processed_dir=args.processed_dir,
                      designs=curr_designs,
                      logic=args.logic,
                      recipes=args.recipes,
                      dim_input=args.dim_input,
                      dim_hidden=args.dim_hidden,
                      epoch_size=args.epoch_size,
                      batch_size=args.batch_size,
                      workspace=args.workspace)
    trainer.run()