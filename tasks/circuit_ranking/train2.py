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

from dataset2 import RankingDataset2
from net import CircuitRankNet, CircuitRankNet2

from datetime import datetime

from src.utils.feature import padding_feature_to
from src.utils.plot import plot_curve, plot_dual_curve

from torch_geometric.loader import DataLoader

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]

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
        
        self.acc_bar = 0.5
        
        self.current_time = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.workspace = os.path.join(workspace, self.current_time)
        os.makedirs(self.workspace, exist_ok=True)
        self.log_file = os.path.join(self.workspace, "log.txt")
        self.success_file = os.path.join(self.workspace, "success.txt")
        self.fail_file = os.path.join(self.workspace, "fail.txt")

        # load the dataset first
        self.dataset = RankingDataset2(root_openlsd=root_openlsd, processed_dir=processed_dir, designs=designs, logics=logics, recipes=recipes)
        # set the model
        # self.model = CircuitRankNet(self.dim_input, self.dim_hidden).to(device)
        self.model = CircuitRankNet2(self.dim_input, self.dim_hidden).to(device)
        # self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        
        print("Configuration of Circuit Ranking:")
        print(f"root_openlsd: {self.root_openlsd}")
        print(f"processed_dir: {self.processed_dir}")
        print(f"designs: {self.designs}")
        print(f"logics: {self.logics}")
        print(f"recipes: {self.recipes}")
        print(f"dim_input: {self.dim_input}")
        print(f"dim_hidden: {self.dim_hidden}")
        print(f"epoch: {self.epoch}")
        print(f"optimizer: SGD")
        print(f"batch_size: {self.batch_size}")
        print(f"workspace: {self.workspace}")
        print(f"device: {device}")
        
        with open(self.log_file, "w") as f:
            f.write("Configuration of Circuit Ranking:\n")
            f.write(f"root_openlsd: {self.root_openlsd}\n")
            f.write(f"processed_dir: {self.processed_dir}\n")
            f.write(f"designs: {self.designs}\n")
            f.write(f"logics: {self.logics}\n")
            f.write(f"recipes: {self.recipes}\n")
            f.write(f"dim_input: {self.dim_input}\n")
            f.write(f"dim_hidden: {self.dim_hidden}\n")
            f.write(f"epoch: {self.epoch}\n")
            f.write(f"optimizer: SGD\n")
            f.write(f"batch_size: {self.batch_size}\n")
        
        with open(self.success_file, "w") as f:
            f.write("Success:\n")
            
        with open(self.fail_file, "w") as f:
            f.write("Fail:\n")
            
    def preproccess_graph(self):
        for data in self.dataset:
            data["graph_0"] = padding_feature_to(data["graph_0"], self.dim_input)
            data["graph_1"] = padding_feature_to(data["graph_1"], self.dim_input)
    
    def run(self):
        self.preproccess_graph()

        dataset_train, dataset_test, test_desgin_recipes = self.dataset.split_train_test(0.8)
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)
            
        torch.manual_seed(12345)
        total_loss_train, total_acc_train, total_precision_train, total_recall_train, total_f1_train = [], [], [], [], []
        total_loss_test, total_acc_test, total_precision_test, total_recall_test, total_f1_test = [], [], [], [], []
        
        print("Start training:")
        
        with open(self.log_file, "a") as f:
            f.write("Start training:\n")
        
        for epoch in range(self.epoch):
            loss_train, acc_train, precision_train, recall_train, f1_train = self.train(dataloader_train)
            loss_test, acc_test, precision_test, recall_test, f1_test = self.eval(dataloader_test)

            # test current improvement for each design and recipes
            with open(self.success_file, "a") as f:
                f.write(f"Epoch{epoch}\n")
            with open(self.fail_file, "a") as f:
                f.write(f"Epoch{epoch}\n")          
            self.eval_group(test_desgin_recipes)
            
            
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
            with open(self.log_file, "a") as f:
                f.write(f"Epoch: {epoch:03d}, Loss: {loss_train:.4f} ({loss_test:.4f}), Acc: {acc_train:.4f} ({acc_test:.4f}), Precision: {precision_train:.4f} ({precision_test:.4f}), Recall: {recall_train:.4f} ({recall_test:.4f}), F1: {f1_train:.4f} ({f1_test:.4f})\n")

            
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

    def logic_embedding(self, logic: List) -> torch.Tensor:
        logic_embed = torch.zeros(len(logic), len(Logics), dtype=torch.float32)
        for i, l in enumerate(logic):
            logic_embed[i, Logics.index(l)] = 1.0
        return logic_embed
    
    def train(self, dataloader:DataLoader):
        self.model.train()
        total_loss = 0
        total_trues = 0
        cnt = 0

        tp = 0
        fp = 0
        tn = 0
        fn = 0
            
        for data in tqdm(dataloader, desc="training"):
            self.optimizer.zero_grad()
            graph_0 = data['graph_0'].to(device)
            graph_1 = data['graph_1'].to(device)
            logic_0 = data['logic_0']
            logic_1 = data['logic_1']
            logic0_embed = self.logic_embedding(logic_0).to(device)
            logic1_embed = self.logic_embedding(logic_1).to(device)
            target = data['label'].float().to(device)
            output = self.model(graph_0, logic0_embed, graph_1, logic1_embed)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            cnt += len(target)
            
            predictions = torch.abs( target-output ) < self.acc_bar
            
            trues = torch.sum(predictions).item()

            tp += torch.sum((predictions == 1) & (target == 1)).item()
            tn += torch.sum((predictions == 0) & (target == 0)).item()
            fp += torch.sum((predictions == 0) & (target == 1)).item()
            fn += torch.sum((predictions == 1) & (target == 0)).item()
            
            total_trues += trues
        
        acc = total_trues / cnt
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        
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
        
        with torch.no_grad():
            for data in tqdm(dataloader, desc="evaluating"):
                graph_0 = data['graph_0'].to(device)
                graph_1 = data['graph_1'].to(device)
                logic_0 = data['logic_0']
                logic_1 = data['logic_1']
                logic0_embed = self.logic_embedding(logic_0).to(device)
                logic1_embed = self.logic_embedding(logic_1).to(device)
                target = data['label'].float().to(device)
                output = self.model(graph_0, logic0_embed, graph_1, logic1_embed)
                loss = F.binary_cross_entropy(output, target)
                total_loss += loss.item()
                
                cnt += len(target)
                
                predictions = torch.abs( target-output ) < self.acc_bar
                
                trues = torch.sum(predictions).item()

                tp += torch.sum((predictions == 1) & (target == 1)).item()
                tn += torch.sum((predictions == 0) & (target == 0)).item()
                fp += torch.sum((predictions == 0) & (target == 1)).item()
                fn += torch.sum((predictions == 1) & (target == 0)).item()
                
                total_trues += trues
            
            acc = total_trues / cnt
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
        
        return total_loss, acc, precision, recall, f1_score

    def eval_group(self, design_recipes):
        self.model.eval()
        
        with torch.no_grad():
            count_success = 0
            count_fail = 0
            for design, recipes in design_recipes.items():
                for recipe, group in recipes.items():
                    datas = {}
                    for data in group:
                        graph_0 = data['graph_0'].to(device)
                        graph_1 = data['graph_1'].to(device)
                        logic_0 = data['logic_0']
                        logic_1 = data['logic_1']
                        timing_0 = data['timing_0']
                        timing_1 = data['timing_1']
                        area_0 = data['area_0']
                        area_1 = data['area_1']
                        if logic_0 not in datas:
                            datas[logic_0] = []
                        if logic_1 not in datas:
                            datas[logic_1] = []
                        datas[logic_0] = [graph_0, timing_0, area_0]
                        datas[logic_1] = [graph_1, timing_1, area_1]
                    
                    if "aig" not in datas:
                        continue
                        
                    # real ranking
                    sorted_datas = sorted(datas.items(), key=lambda x: (x[1][1], x[1][2]) )

                    sorted_by_model = list(datas.items())
                    
                    n = len(sorted_by_model)
                    for i in range(n):
                        for j in range(n - i - 1):
                            logic_i, (graph_i, timing_i, area_i) = sorted_by_model[j]
                            logic_j, (graph_j, timing_j, area_j) = sorted_by_model[j + 1]
                            logici_embed = self.logic_embedding([logic_i]).to(device)
                            logicj_embed = self.logic_embedding([logic_j]).to(device)
                            output = self.model(graph_i, logici_embed, graph_j, logicj_embed).item()
                            if output == 1:
                                sorted_by_model[j], sorted_by_model[j + 1] = sorted_by_model[j + 1], sorted_by_model[j]
                    

                    # checking the best graph is better than the AIG one by the prediction
                    real_best_logic, (real_best_graph, real_best_timing, real_best_area) = sorted_datas[0]
                    pred_best_logic, (pred_best_graph, pred_best_timing, pred_best_area) = sorted_by_model[0]
                    aig_graph, aig_timing, aig_area = datas["aig"]
                    
                    if real_best_logic == "aig":
                        continue
                    
                    logic_real_embed = self.logic_embedding([real_best_logic]).to(device)
                    logic_pred_embed = self.logic_embedding([pred_best_logic]).to(device)
                    target = torch.tensor([0]).float().to(device)
                    
                    # 这里的顺序是对的吗？
                    output = self.model(pred_best_graph.to(device), logic_pred_embed, real_best_graph.to(device), logic_real_embed).item()
                    
                    is_success = torch.abs( target-output ) < self.acc_bar
                    
                    
                    timing_improve = (aig_timing-pred_best_timing)/aig_timing
                    area_improve = (aig_area - pred_best_area)/aig_area
                    if is_success:
                        assert(torch.abs( target-output ) < self.acc_bar)
                        count_success += 1
                        with open(self.success_file, "a") as f:
                            f.write(f"design {design}, recipe {recipe}\n")
                            f.write(f'logic_0: {pred_best_logic}, logic_1: "aig", timing_0: {pred_best_timing}, timing_1: {aig_timing}, area_0: {pred_best_area}, area_1: {aig_area}\n')
                            f.write(f'best_logic: {real_best_logic}, timing_best: {real_best_timing}, area_best: {real_best_area}\n')
                            f.write(f'the current improvement is {timing_improve:0.4f} in timing and {area_improve:0.4f} in area\n')
                    else:
                        assert(torch.abs( target-output ) >= self.acc_bar)
                        count_fail += 1
                        with open(self.fail_file, "a") as f:
                            f.write(f"design {design}, recipe {recipe}\n")
                            f.write(f'logic_0: {pred_best_logic}, logic_1: "aig", timing_0: {pred_best_timing}, timing_1: {aig_timing}, area_0: {pred_best_area}, area_1: {aig_area}\n')
                            f.write(f'best_logic: {real_best_logic}, timing_best: {real_best_timing}, area_best: {real_best_area}\n')
                            f.write(f'the current improvement is {timing_improve:0.4f} in timing and {area_improve:0.4f} in area\n')
                        
            count = count_success + count_fail
            rate = count_success / count
            print(f"ranking success rate: {rate} = {count_success}/{count}")
            with open(self.log_file, "a") as f:
                f.write(f"ranking success rate: {rate} = {count_success}/{count}\n")

    def visualize(self, dataloader:DataLoader, title:str, save_path:str):
        self.model.eval()
        
        total_acc = []
        with torch.no_grad():
            for data in dataloader:
                graph_0 = data['graph_0'].to(device)
                graph_1 = data['graph_1'].to(device)
                
                logic_0 = data['logic_0']
                logic_1 = data['logic_1']
                logic0_embed = self.logic_embedding(logic_0).to(device)
                logic1_embed = self.logic_embedding(logic_1).to(device)
                
                target = data['label'].float().to(device)
                output = self.model(graph_0, logic0_embed, graph_1, logic1_embed)

                predictions = torch.abs( target-output ) < self.acc_bar
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
    parser.add_argument('--dim_input', type=int, default=128, help='the dimenssion of the feature size for each node')
    parser.add_argument('--dim_hidden', type=int, default=256, help='the dimenssion of the feature size for each node')
    parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the dataloader')
    parser.add_argument('--workspace', type=str, required=True, help='the path of the workspace to store the results')
    
    args = parser.parse_args()

    curr_logics = ["aig", "oig", "xag", "primary", "mig", "gtg"]

    # curr_designs = [
    #     "ctrl",
    #     "router",
    #     "int2float",
    #     "ss_pcm",
    #     "usb_phy",
    #     "sasc",
    #     "cavlc",
    #     "simple_spi",
    #     "priority",
    #     "i2c",
    #     ]
    
    curr_designs = [
        "adder",
        "bar",
        "max",
        "sin",
        "iir",
        "cavlc",
        "des3_area",
        "systemcdes",
        "ctrl",
        "priority",
        "router",
        "steppermotordrive",
        "spi",
        "ss_pcm",
        "usb_phy",
        "sasc",
        "int2float",
        "fir",
        "i2c",
        "wb_dma",  
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