import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import gzip

CmdPool = [
    "refactor",
    "refactor -l",
    "refactor -z",
    "refactor -l -z",
    "rewrite",
    "rewrite -l",
    "rewrite -z",
    "rewrite -l -z",
    "resub",
    "resub -l",
    "resub -z",
    "resub -l -z",
    "balance",
    "balance",
    "balance",
    "balance",
]

def load_seq(file):
    cmds = []
    if file.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(file, 'rt') as f:
        for line in f:
            array = line.strip().split(';')
            array = [item.strip() for item in array if item.strip()]
            cmds.extend(array)
    for seq in cmds:
        if seq not in CmdPool:
            raise ValueError(f"Invalid command: {seq}")
    seq = ';'.join(cmds)
    return seq

if __name__ == "__main__":
    """_summary_
    """
    seq_file = sys.argv[1]
    seq = load_seq(seq_file)
    print(seq)