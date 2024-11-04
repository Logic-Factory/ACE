import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim import SGD

from dataset import RankingDataset
from net import CircuitRankNet

from datetime import datetime

from src.utils.feature import padding_feature_to
from src.utils.plot import plot_curve

from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, root_openlsd:str, processed_dir:str, designs:List[str], logics: List[str], recipes:int, feature_size:int, epoch_size:int, batch_size:int, workspace:str):
        self.root_openlsd = root_openlsd
        self.processed_dir = processed_dir
        self.designs = designs
        self.logics = logics
        self.recipes = recipes
        
        self.feature_size = feature_size
        self.epoch = epoch_size
        self.batch_size = batch_size
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, self.current_time)

        # load the dataset first
        self.dataset = RankingDataset(root_openlsd=root_openlsd, processed_dir=processed_dir, designs=designs, logics=logics, recipes=recipes)
        # set the model
        self.model = CircuitRankNet(feature_size).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        print("Configuration of Circuit Ranking:")
        print(f"root_openlsd: {self.root_openlsd}")
        print(f"processed_dir: {self.processed_dir}")
        print(f"designs: {self.designs}")
        print(f"logics: {self.logics}")
        print(f"recipes: {self.recipes}")
        print(f"feature_size: {self.feature_size}")
        print(f"epoch: {self.epoch}")
        print(f"batch_size: {self.batch_size}")
        print(f"workspace: {self.workspace}")
        print(f"device: {device}")
        
    def preproccess_graph(self):
        for data in self.dataset:
            data["graph_0"] = padding_feature_to(data["graph_0"], self.feature_size)
            data["graph_1"] = padding_feature_to(data["graph_1"], self.feature_size)
    
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
        
    def train(self, dataloader:DataLoader):
        self.model.train()
        total_loss = 0
        total_acc = 0
        for data in dataloader:
            self.optimizer.zero_grad()
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            target = graph_0.y
            output = self.model(graph_0, graph_1)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if abs( target.item() - output.item() ) < 0.2:
                total_acc += 1
        total_acc /= len(dataloader)
        return total_loss, total_acc
    
    def eval(self, dataloader:DataLoader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        for data in dataloader:            
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            target = graph_0.y.float().to(device)
            output = self.model(graph_0, graph_1)
            loss = F.binary_cross_entropy(output, target)
            total_loss += loss.item()

            if abs( target.item() - output.item() ) <  0.2:
                total_acc += 1
        total_acc /= len(dataloader)
        
        return total_loss, total_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_openlsd', type=str, required=True, help='the path of the datapath')
    parser.add_argument('--processed_dir', type=str, help='the dimenssion of the feature size for each node')
    parser.add_argument('--recipes', type=int, default=50, help='the extracted recipe size for each design')
    parser.add_argument('--feature_size', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]
    curr_designs = [
        "ctrl",
        "steppermotordrive"
        ]
    
    trainer = Trainer(root_openlsd=args.root_openlsd,
                      processed_dir=args.processed_dir,
                      designs=curr_designs,
                      logics=curr_logics,
                      recipes=args.recipes,
                      feature_size=args.feature_size,
                      epoch_size=args.epoch_size,
                      batch_size=args.batch_size,
                      workspace=args.workspace)
    trainer.run()