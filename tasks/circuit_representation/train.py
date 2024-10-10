import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from typing import List
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim import SGD

from dataset import RepresentationDataset
from net import CircuitRankNet

from datetime import datetime

from src.utils.feature import padding_feature_to

from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, root_openlsd:str, recipe_size:int, curr_designs:List[str], processed_dir:str, feature_size:int, epoch:int, batch_size:int, workspace:str):
        self.curr_designs = curr_designs
        self.processed_dir = processed_dir
        self.feature_size = feature_size
        self.epoch = epoch
        self.batch_size = batch_size
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, self.current_time)

        # load the dataset first
        self.dataset = RepresentationDataset(root_openlsd, recipe_size, curr_designs, processed_dir)
        self.model = CircuitRankNet(feature_size).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
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
        return total_loss, total_acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_openlsd', type=str, required=True, help='the path of the datapath')
    parser.add_argument('--recipe_size', type=int, default=50, help='the extracted recipe size for each design')
    parser.add_argument('--processed_dir', type=str, help='the dimenssion of the feature size for each node')
    parser.add_argument('--feature_size', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_designs = ["i2c", "priority", "ss_pcm", "tv80"]
    
    trainer = Trainer(args.root_openlsd, args.recipe_size, curr_designs, args.processed_dir, args.feature_size, args.epoch_size, args.batch_size, args.workspace)
    trainer.run()