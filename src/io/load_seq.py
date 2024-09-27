import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import gzip
import zstandard as zstd

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
    if file.endswith('.zst'):
        with open(file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                line = reader.read().decode('utf-8')
                array = line.strip().split(';')
                array = [item.strip() for item in array if item.strip()]
                cmds.extend(array)
    elif file.endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            for line in f:
                array = line.strip().split(';')
                array = [item.strip() for item in array if item.strip()]
                cmds.extend(array)
    else:
        with open(file, 'rt') as f:
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