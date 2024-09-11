import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import re
import pandas as pd
import argparse
import torch
import shutil
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

node_types = {
    "pi" : 0,
    "po" : 1,
    "and" : 2,
}

edge_types = {
    "not" : 0,
    "buf" : 1,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Process GraphML files and create dataset.")
    parser.add_argument("--dataset_dir", type=str, required=False,default='../../data/dataset', help="Root directory for storing processed dataset.")
    parser.add_argument("--graphml_dir", type=str, required=False, default='../../data/graphml', help="Directory containing the GraphML files.")
    parser.add_argument('--design_class', type=str, default='comb', choices=['comb','core'], help='Class of the AIG benchmark')
    return parser.parse_args()

def nodes_analyse_num(G, node_types_of_interest ): 
    
    nodes = {node_type: 0 for node_type in node_types_of_interest}

    node_type_count = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_type = data.get('type')
        if node_type in nodes:  
            nodes[node_type] += 1  

    return nodes

def edges_analyse_num(G,edge_types_of_interest):
    
    edges = {edge_type: 0 for edge_type in edge_types_of_interest}

    # 遍历图中的所有边和它们的属性
    for u, v, attr in G.edges(data=True):
        edge_type = attr.get('type')
        if edge_type in edges:  
            edges[edge_type] += 1  

    return edges

def add_aig_label(data, id, stats_path):
    df = pd.read_csv(stats_path)
    data.area_norm = torch.tensor([df.loc[id, 'areas_norm']],dtype=torch.float)
    data.delay_norm = torch.tensor([df.loc[id, 'delays_norm']],dtype=torch.float)
    return data

def Graphml_to_Data(G, node_types, edge_types):
    node_attrs = nx.get_node_attributes(G, 'type')
    edge_attrs = nx.get_edge_attributes(G, 'type')

    num_node_types = len(node_types)  
    x = torch.zeros((len(G.nodes)+1, num_node_types), dtype=torch.long)  

    for node, attr in node_attrs.items():
        if attr == 'zero':
            continue  
        type_idx = node_types[attr]  
        x[int(node), type_idx] = 1 

    num_edge_types = len(edge_types)  
    edge_attr = torch.zeros((len(G.edges()), num_edge_types), dtype=torch.long)
    for edge, attr in edge_attrs.items():
        type_idx = edge_types[attr]  
        src, tgt = edge
        idx = int(G.edges[src, tgt]['id'])  
        edge_attr[idx, type_idx] = 1

    if not all(isinstance(node, int) for node in G.nodes()):
        node_map = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, node_map)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def process_graphml_file(folder_path, file_name):
    csv_path = os.path.join(folder_path, 'stats/recipes.csv')
    graphml_path = os.path.join(folder_path, file_name)
    G = nx.read_graphml(graphml_path)
    node = nodes_analyse_num(G, node_types)
    edge = edges_analyse_num(G, edge_types)
    data = Graphml_to_Data(G, node_types, edge_types)
    data.pi_num = torch.tensor([node['pi']],dtype=torch.int)
    data.po_num = torch.tensor([node['po']],dtype=torch.int)
    data.and_num = torch.tensor([node['and']],dtype=torch.int)
    data.not_num = torch.tensor([edge['not']],dtype=torch.int)
    data.buf_num = torch.tensor([edge['buf']],dtype=torch.int)
    match = re.search(r"(\d+)(?=\.graphml)", file_name)
    if match:
        i = int(match.group())  # 提取数字并转换为整数
    data = add_aig_label(data, i, csv_path)
    return data


class GraphMLDataset(InMemoryDataset):
    def __init__(self, root, graphml_dir, transform=None, pre_transform=None, force_reload=False, empty=False):
        self.root = root
        self.graphml_dir = graphml_dir
        self.transform, self.pre_transform = transform, pre_transform

        super(GraphMLDataset, self).__init__(root, transform, pre_transform)

        if not force_reload and self._check_processed_files_exist():
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif not empty:
            self.process()
            torch.save((self.data, self.slices), self.processed_paths[0])

    def _check_processed_files_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'graphmldata.pt'
    
    def process(self):
        data_list = []
        folder_paths = [os.path.join(self.graphml_dir, d) for d in os.listdir(self.graphml_dir) if os.path.isdir(os.path.join(self.graphml_dir, d))]

        total_files = sum(len(os.listdir(os.path.join(self.graphml_dir, d))) for d in folder_paths if d.endswith('.graphml'))
        with tqdm(total=total_files, desc='Processing files') as pbar:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for folder_path in folder_paths:

                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.graphml'):
                            future = executor.submit(process_graphml_file, folder_path, file_name)
                            futures.append(future)
                for future in as_completed(futures):
                    data = future.result()
                    data_list.append(data)
                    pbar.update(1) 


        print(f"Total data objects created: {len(data_list)}")

        random.shuffle(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def main(args):
    dataset_dir = os.path.join(args.dataset_dir, args.design_class)
    graphml_dir = os.path.join(args.graphml_dir, args.design_class)
    dataset = GraphMLDataset(dataset_dir, graphml_dir, transform=None, pre_transform=None)

if __name__ == '__main__':
    args = parse_args()
    main(args)