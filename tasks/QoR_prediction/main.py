import os
import sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)
import argparse
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .graphml2data import GraphMLDataset
from model import GCN

criterion = torch.nn.MSELoss()

def train(model,device,dataloader,optimizer):
    model.train()
    total_mse = 0
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
        batch = batch.to(device)
        lbl = batch.area_norm.reshape(-1, 1)
        optimizer.zero_grad()
        batch.x = batch.x.float()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred,lbl)
        total_mse += loss.item()
        loss.backward()
        optimizer.step()
    return total_mse/len(dataloader)


def test(model, device, dataloader):
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            batch = batch.to(device)
            lbl = batch.area_norm.reshape(-1, 1)
            batch.x = batch.x.float()
            pred = model(batch.x, batch.edge_index, batch.batch)
            mseVal = criterion(pred, lbl)
            total_mse += mseVal.item()
    return total_mse/len(dataloader)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on Synthesis Task Pytorch Geometric')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train (default: 80)')
    parser.add_argument('--design_class', type=str, default="comb_1w", help='Dataset name')
    parser.add_argument('--graphml_dir', type=str, default="../../data/graphml", help='GraphML directory containing graphml files')
    parser.add_argument('--dataset_dir', type=str, default="../../data/dataset", help='Dataset directory containing processed dataset ')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument('--target', type=str,  default="delay", help='Target label (area/delay), default:"area"')
    parser.add_argument('--result_dir', type=str, default="./results", help='Directory to save results')
    args = parser.parse_args()
    graphml_dir = os.path.join(args.graphml_dir, args.design_class)
    dataset_dir = os.path.join(args.dataset_dir, args.design_class)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.design_class, args.target)
    result_dir = os.path.join(args.result_dir, args.design_class, args.target)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_csv_dir = os.path.join(result_dir, "result.csv")
    if not os.path.exists(result_csv_dir):
        with open(result_csv_dir, 'w', newline='') as result_csv_file:
            csv_writer = csv.writer(result_csv_file)
            csv_writer.writerow(['epoch', 'train mse','test mse','best mse'])

    data = GraphMLDataset(dataset_dir, graphml_dir)

    len_D = len(data)
    train_size = int(len_D * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


    nodeEmbeddingDim =256

    model = GCN(nodeEmbeddingDim)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_curve = []
    train_loss = []
    testLossOpt = 1
    besttestEpoch = 1

    for ep in range(1, args.epochs + 1):
        print("\nEpoch [{}/{}]".format(ep, args.epochs))
        print("\nTraining..")
        trainLoss = train(model, device, train_dl, optimizer)
        print("\nEvaluation..")
        testLoss = test(model, device, test_dl)
        if ep > 1:
            if testLossOpt > testLoss:
                testLossOpt = testLoss
                besttestEpoch = ep
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'gcn-epoch-{}-test_loss-{:.3f}.pt'.format(besttestEpoch, testLossOpt)))
        else:
            testLossOpt = testLoss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'gcn-epoch-{}-test_loss-{:.3f}.pt'.format(besttestEpoch, testLossOpt)))
        print({'Train loss': trainLoss,'Test loss': testLoss})
        test_curve.append(testLoss)
        train_loss.append(trainLoss)
        scheduler.step(testLoss)

    
        with open(result_csv_dir, "a", newline='') as result_csv_file:
            csv_writer = csv.writer(result_csv_file)
            csv_writer.writerow([ep, trainLoss, testLoss,testLossOpt])
    
if __name__ == '__main__':
    main()