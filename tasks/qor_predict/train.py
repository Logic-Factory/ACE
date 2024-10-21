import os
import sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

from net import *
from dataset import QoR_Dataset

from typing import List
from datetime import datetime
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.plot import plot_curve, plot_2d_dots

criterion = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_mape(actual, forecast):

    actual = actual[actual != 0]
    forecast = forecast[forecast != 0]
    
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return mape


class Trainer(object):
    def __init__(self, root_openlsd:str, recipe_size:int, curr_designs:List[str], processed_dir:str, logic:str, dim_input:int, dim_hidden:int, epoch:int, batch_size:int,target:str, workspace:str,nodeEmbeddingDim:int, synthEncodingDim:int, lr:float):
        self.curr_designs = curr_designs
        self.root_openlsd = root_openlsd
        self.processed_dir = processed_dir
        self.logic = logic
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.epoch = epoch
        self.batch_size = batch_size
        self.target = target
        self.nodeEmbeddingDim = nodeEmbeddingDim
        self.synthEncodingDim = synthEncodingDim
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, logic, self.current_time)
        os.makedirs(self.workspace, exist_ok=True)
        
        # load the dataset first
        self.dataset = QoR_Dataset(root_openlsd, recipe_size, curr_designs, processed_dir, logic, target)
        self.node_encoder = NodeEncoder(emb_dim=self.nodeEmbeddingDim)
        self.synthesis_encoder = SynthFlowEncoder(emb_dim=self.synthEncodingDim)
        self.model = SynthNet(
            node_encoder=self.node_encoder,
            synth_encoder=self.synthesis_encoder,
            n_classes=1,
            synth_input_dim=self.synthEncodingDim,
            node_input_dim=self.nodeEmbeddingDim,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", verbose=True)
    
    def run(self):
        dataset_train, dataset_test = self.dataset.split_train_test(0.8)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        best_score = 999
        torch.manual_seed(12345)
        total_loss_train = []
        total_loss_test = []
        for epoch in range(self.epoch):
            loss_train = self.train(dataloader_train)
            loss_test = self.eval(dataloader_test)
            total_loss_train.append(loss_train)
            total_loss_test.append(loss_test)
            
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.4f} ({loss_test:.4f})')
            if loss_test < best_score and epoch > 1 and loss_train < 0.7:
                best_score = loss_test
                torch.save(self.model.state_dict(), os.path.join(self.workspace, f"best_model_{best_score:.3f}.pt"))
            
            plot_curve(lists= [total_loss_train], labels=[], title="train loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace, "loss_train.pdf"))
            plot_curve(lists= [total_loss_test], labels=[], title="test loss curve", x_label="epoch", y_label="loss", save_path = os.path.join(self.workspace,"loss_test.pdf"))
        
        self.model.load_state_dict(torch.load(os.path.join(self.workspace, f"best_model_{best_score:.3f}.pt")))
        self.eval_plot(dataloader_test)

    def train(self, dataloader:DataLoader):
        
        self.model.train()
        total_mse = 0
        for _, batch in enumerate(tqdm(dataloader, desc="Train",file=sys.stdout)):
            batch = batch.to(device)
            lbl = batch.target.reshape(-1, 1)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = criterion(pred,lbl)
            total_mse += loss.item()
            loss.backward()
            self.optimizer.step()
        return total_mse/len(dataloader)

    def eval(self, dataloader:DataLoader):
        self.model.eval()
        total_mse = 0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc="Test",file=sys.stdout)):
                batch = batch.to(device)
                lbl = batch.target.reshape(-1, 1)
                pred = self.model(batch)
                mseVal = criterion(pred, lbl)
                # print(f'pred: {pred}, lbl: {lbl}')
                total_mse += mseVal.item()
        return total_mse/len(dataloader)
    
    def eval_plot(self, dataloader:DataLoader):
        self.model.eval()

        batch_data = []
        design_dir = os.path.join(self.root_openlsd,'design_data.json')
        with open(design_dir, 'r') as file:
            design_data = json.load(file)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc="eval_plot",file=sys.stdout)):
                batch = batch.to(device)
                lbl = batch.target.reshape(-1, 1)
                name = batch.name
                pred = self.model(batch)
                predArray = pred.view(-1,1).detach().cpu().numpy()
                lblArray = lbl.view(-1,1).detach().cpu().numpy()
                batch_data.append([name, predArray, lblArray])
        flattened_data = []

        for entry in batch_data:
            design_names = entry[0]
            pred_array = entry[1].flatten() 
            lbl_array = entry[2].flatten()    

            flattened_data.append([design_names, pred_array, lbl_array])
        act = {}
        pred = {}
        for name, predArray, lblArray in flattened_data:
            for i in range(len(name)):
                key = f'{name[i]}'
                if key not in act:
                    act[key] = []
                act[key].append(lblArray[i] * design_data[name[i]]['std'] + design_data[name[i]]['mean'])
                key = f'{name[i]}'
                if key not in pred:
                    pred[key] = []
                pred[key].append(predArray[i]*design_data[name[i]]['std'] + design_data[name[i]]['mean'])
        for key in act:
            mape = calculate_mape(pred[key], act[key])
            plot_2d_dots(x_list=pred[key], y_list=act[key], title=f"MAPE{mape:.3f}%", x_label="prediction", y_label="actual", save_path = os.path.join(self.workspace, f"{key}.pdf"))
                

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_openlsd', type=str, required=True,default = '', help='the path of the datapath')
    parser.add_argument('--recipe_size', type=int, default=500, help='the extracted recipe size for each design')
    parser.add_argument('--processed_dir', type=str,default='', help='the dimenssion of the feature size for each node')
    parser.add_argument('--logic', type=str, default="abc", help='the logic type of the selected dataset')
    parser.add_argument('--dim_input', type=int, default=64, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=128, help='the dimension of the hidden layer')
    parser.add_argument('--epoch_size', type=int, default=50, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size of the dataloader')
    parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate of the optimizer')
    parser.add_argument('--nodeEmbeddingDim', type=int, default=32, help='the dimension of the node embedding')
    parser.add_argument('--synthEncodingDim', type=int, default=32, help='the dimension of the synthesis encoding')
    parser.add_argument('--target', type=str, default='area', choices=['area','delay'],help='the target to be predicted')
    parser.add_argument('--workspace', type=str, required=True,default='', help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_designs = ["i2c", "ss_pcm"]

    trainer = Trainer(args.root_openlsd, args.recipe_size, curr_designs, args.processed_dir, args.logic, args.dim_input, args.dim_hidden, args.epoch_size, args.batch_size, args.target, args.workspace, args.nodeEmbeddingDim, args.synthEncodingDim, args.lr)
    trainer.run()
    

    
    
if __name__ == '__main__':
    main()