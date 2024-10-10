import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import torch
from torch_geometric.data import Data

def padding_feature_to(graph:Data, feature_size:int):
    """_summary_

    Args:
        graph (Data): _description_
        feature_size (int): _description_

    Returns:
        _type_: _description_
    """
    pad_size = feature_size - graph.x.shape[1]
    assert pad_size >= 0, "original feature size is too large"
    graph.x = graph.x.to(torch.float)
    additional_features = torch.randn(len(graph.x), pad_size)
    graph.x = torch.cat((graph.x, additional_features), dim=1)
    return graph
  

def padding_feature_to_nochange(x, feature_size:int):
    """_summary_

    Args:
        graph (Data): _description_
        feature_size (int): _description_

    Returns:
        _type_: _description_
    """
    pad_size = feature_size - x.shape[1]
    assert pad_size >= 0, "original feature size is too large"
    x = x.to(torch.float)
    additional_features = torch.randn(len(x), pad_size)
    x = torch.cat((x, additional_features), dim=1)
    return x