import prune_ckpt
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='prune all ckpt')
parser.add_argument('--delete', type=bool, default=None, help='delete the 11gb files')
args = parser.parse_args()
ckpt = args.delete

# path to logs folder
logs_path = "logs"

# resurcively search logs folder for ckpt files and prune them
for path in glob.glob(logs_path + "/**/*.ckpt", recursive=True):
    path_here = os.path.basename(path)
    
    os.rename(path, path_here)

    prune_ckpt.prune_it(path_here, keep_only_ema=False)
    
    if args.delete:
        os.remove(path_here)