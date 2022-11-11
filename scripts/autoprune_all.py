import prune_ckpt
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='prune all ckpt')
parser.add_argument("--delete", "-d", type=bool, nargs="?", const=True, default=False, help="delete the 11gb files")
args = parser.parse_args()

logs_path = "logs"

print("Copying all ckpt files from logs folder to root and pruning, with delete 11gb files option: ", args.delete)

file_list = glob.glob(logs_path + "/**/*.ckpt", recursive=True)

if not file_list:
    print("No ckpt files found")
else:
    for path in file_list:
        path_here = os.path.basename(path)

        print(f"moving {path_here}")
        
        try:
            os.rename(path, path_here)

            prune_ckpt.prune_it(path_here, keep_only_ema=False)
        
            if args.delete:
                os.remove(path_here)
        except Exception as e:
            print(f"Error on file {path} with error {e}")