import os.path
import os.path
import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import numpy as np
import pandas as pd

from src.utils.plot import plot_curve2

def plot_acc():
    data_batch_no = "/home/niliwei/Logic-Factory/ACE/tasks/circuit_classification/result2/aig/2025_0121_200407/data.csv"

    # load the data
    data = pd.read_csv(data_batch_no)
 
    # compute the ave
    test_acc_aig = data["test_acc"]
    test_acc_oig = data["test_acc_oig"]
    test_acc_xag = data["test_acc_xag"]
    test_acc_mig = data["test_acc_mig"]
    test_acc_primary = data["test_acc_primar"]
    test_acc_gtg = data["test_acc_gtg"]
       
    
    # plot the curves
    labels = ["AIG", "OIG", "XAG", "MIG", "PRIMARY", "GTG"]
    plot_curve2(lists=[test_acc_aig, test_acc_oig, test_acc_xag, test_acc_mig, test_acc_primary, test_acc_gtg], labels=labels, x_label="Epoch", y_label="Accuracy", title="", save_path="test_bn_acc.pdf", isLegend=True)
plot_acc()