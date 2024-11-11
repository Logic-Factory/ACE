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
from src.utils.plot import plot_curve, plot_dual_curve

from torch_geometric.loader import DataLoader

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, root_openlsd:str, processed_dir:str, designs:List[str], logics: List[str], recipes:int, dim_input:int, dim_hidden:int, epoch_size:int, batch_size:int, workspace:str):
        self.root_openlsd = root_openlsd
        self.processed_dir = processed_dir
        self.designs = designs
        self.logics = logics
        self.recipes = recipes
        
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.epoch = epoch_size
        self.batch_size = batch_size
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, self.current_time)

        # load the dataset first
        self.dataset = RankingDataset(root_openlsd=root_openlsd, processed_dir=processed_dir, designs=designs, logics=logics, recipes=recipes)
        # set the model
        self.model = CircuitRankNet(self.dim_input, self.dim_hidden).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        print("Configuration of Circuit Ranking:")
        print(f"root_openlsd: {self.root_openlsd}")
        print(f"processed_dir: {self.processed_dir}")
        print(f"designs: {self.designs}")
        print(f"logics: {self.logics}")
        print(f"recipes: {self.recipes}")
        print(f"dim_input: {self.dim_input}")
        print(f"dim_hidden: {self.dim_hidden}")
        print(f"epoch: {self.epoch}")
        print(f"batch_size: {self.batch_size}")
        print(f"workspace: {self.workspace}")
        print(f"device: {device}")
        
    def preproccess_graph(self):
        for data in self.dataset:
            data["graph_0"] = padding_feature_to(data["graph_0"], self.dim_input)
            data["graph_1"] = padding_feature_to(data["graph_1"], self.dim_input)
    
    def run(self):
        self.preproccess_graph()

        dataset_train, dataset_test = self.dataset.split_train_test(0.7)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)
            
        torch.manual_seed(12345)
        total_loss_train, total_acc_train, total_precision_train, total_recall_train, total_f1_train = [], [], [], [], []
        total_loss_test, total_acc_test, total_precision_test, total_recall_test, total_f1_test = [], [], [], [], []
        
        os.makedirs(self.workspace, exist_ok=True)
        
        print("Start training:")
        for epoch in range(self.epoch):
            loss_train, acc_train, precision_train, recall_train, f1_train = self.train(dataloader_train)
            loss_test, acc_test, precision_test, recall_test, f1_test = self.eval(dataloader_test)
            total_loss_train.append(loss_train)
            total_acc_train.append(acc_train)
            total_precision_train.append(precision_train)
            total_recall_train.append(recall_train)
            total_f1_train.append(f1_train)
            
            total_loss_test.append(loss_test)
            total_acc_test.append(acc_test)
            total_precision_test.append(precision_test)
            total_recall_test.append(recall_test)
            total_f1_test.append(f1_test)
            
            self.visualize(dataloader_train, title="Pair-wise Prediction Distr (Train)", save_path = os.path.join(self.workspace, f"vis_train_{epoch}.pdf"))
            self.visualize(dataloader_test, title="Pair-wise Prediction Distr (Eval)", save_path = os.path.join(self.workspace, f"vis_test_{epoch}.pdf"))
            
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.4f} ({loss_test:.4f}), Acc: {acc_train:.4f} ({acc_test:.4f}), Precision: {precision_train:.4f} ({precision_test:.4f}), Recall: {recall_train:.4f} ({recall_test:.4f}), F1: {f1_train:.4f} ({f1_test:.4f})')
            plot_curve(lists= [total_loss_train], labels=[], title="train loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, "loss_train.pdf"))
            plot_curve(lists= [total_acc_train], labels=[], title="train acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_train.pdf"))
            plot_curve(lists= [total_loss_test], labels=[], title="test loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace,"loss_test.pdf"))
            plot_curve(lists= [total_acc_test], labels=[], title="test acc curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc_test.pdf"))
            plot_curve(lists= [total_acc_train, total_acc_test], labels=["train", "test"], title="Accuracy Curve", x_label="epoch", y_label="acc", save_path = os.path.join(self.workspace, "acc.pdf"))
            plot_curve(lists= [total_loss_train, total_loss_test], labels=["train", "test"], title="Loss Curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace,"loss.pdf"))
            plot_curve(lists= [total_precision_train, total_recall_train, total_f1_train], labels=["precision", "recall", "f1"], title="Training", x_label="epoch", y_label="", save_path = os.path.join(self.workspace,"score_train.pdf"))
            plot_curve(lists= [total_precision_test, total_recall_test, total_f1_test], labels=["precision", "recall", "f1"], title="Evaluation", x_label="epoch", y_label="", save_path = os.path.join(self.workspace,"score_test.pdf"))

            plot_dual_curve(list0=total_acc_train, list1=total_acc_test, title="Accuracy Curve", x_label="epoch", y0_label="Train", y1_label="Test", save_path = os.path.join(self.workspace, "acc_dual.pdf"))
            plot_dual_curve(list0=total_loss_train, list1=total_loss_test, title="Loss Curve", x_label="epoch", y0_label="Train", y1_label="Test", save_path = os.path.join(self.workspace, "loss_dual.pdf"))
            
        torch.save(self.model.state_dict(), os.path.join(self.workspace, "model.pth"))
    
    def train(self, dataloader:DataLoader):
        self.model.train()
        total_loss = 0
        total_trues = 0
        cnt = 0

        tp = 0
        fp = 0
        tn = 0
        fn = 0
            
        for data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            target = data['label'].float().to(device)
            output = self.model(graph_0, graph_1)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            cnt += len(target)
            
            predictions = torch.abs( target-output ) < 0.2
            
            trues = torch.sum(predictions).item()

            tp += torch.sum((predictions == 1) & (target == 1)).item()
            tn += torch.sum((predictions == 0) & (target == 0)).item()
            fp += torch.sum((predictions == 0) & (target == 1)).item()
            fn += torch.sum((predictions == 1) & (target == 0)).item()
            
            total_trues += trues
        
        acc = total_trues / cnt
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        
        return total_loss, acc, precision, recall, f1_score
    
    def eval(self, dataloader:DataLoader):
        self.model.eval()
        total_loss = 0
        total_trues = 0
        cnt = 0

        tp = 0
        fp = 0
        tn = 0
        fn = 0
            
        for data in tqdm(dataloader, desc="Training"):
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            target = data['label'].float().to(device)
            output = self.model(graph_0, graph_1)
            loss = F.binary_cross_entropy(output, target)
            total_loss += loss.item()
            
            cnt += len(target)
            
            predictions = torch.abs( target-output ) < 0.2
            
            trues = torch.sum(predictions).item()

            tp += torch.sum((predictions == 1) & (target == 1)).item()
            tn += torch.sum((predictions == 0) & (target == 0)).item()
            fp += torch.sum((predictions == 0) & (target == 1)).item()
            fn += torch.sum((predictions == 1) & (target == 0)).item()
            
            total_trues += trues
        
        acc = total_trues / cnt
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        
        return total_loss, acc, precision, recall, f1_score

    def visualize(self, dataloader:DataLoader, title:str, save_path:str):
        self.model.eval()
        
        total_acc = []
        for data in dataloader:
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            logic_0 = data['logic_0']
            logic_1 = data['logic_1']
            target = data['label'].float().to(device)
            output = self.model(graph_0, graph_1)
            predictions = torch.abs( target-output ) < 0.1
            total_acc.extend(predictions)

        
        acc_tensor = torch.tensor(total_acc)
        true_count = torch.sum(acc_tensor == 1).item()
        acc = true_count / len(acc_tensor)
        acc *= 100
        
        n = int(torch.ceil(torch.sqrt(torch.tensor(len(acc_tensor)))))
        total_elements = n * n
        padding = total_elements - len(acc_tensor)
        padding_tensor = torch.full((padding,), True)
        padded_acc_tensor = torch.cat((acc_tensor, padding_tensor))
        acc_matrix = padded_acc_tensor.view(n, n)

        # Plotting the boolean matrix as an image
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.imshow(acc_matrix, cmap='gray', interpolation='nearest')
        plt.title(title + f"(Acc={acc:.2f}%)")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_openlsd', type=str, required=True, help='the path of the datapath')
    parser.add_argument('--processed_dir', type=str, help='the dimenssion of the feature size for each node')
    parser.add_argument('--recipes', type=int, default=50, help='the extracted recipe size for each design')
    parser.add_argument('--dim_input', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=128, help='the dimenssion of the feature size for each node')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_logics = ["aig", "xag", "mig", "gtg"]
    curr_designs = [
        "ctrl",
        "router",
        "int2float",
        "ss_pcm",
        "usb_phy",
        "sasc",
        "cavlc",
        "simple_spi",
        "priority",
        "i2c",
        ]
    
    trainer = Trainer(root_openlsd=args.root_openlsd,
                      processed_dir=args.processed_dir,
                      designs=curr_designs,
                      logics=curr_logics,
                      recipes=args.recipes,
                      dim_input=args.dim_input,
                      dim_hidden=args.dim_hidden,
                      epoch_size=args.epoch_size,
                      batch_size=args.batch_size,
                      workspace=args.workspace)
    trainer.run()