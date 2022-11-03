import os

class DataLoader():
    def __init__(self, data_root, quiet=True):
        self.data_root = data_root
        self.image_paths = []
        self.quiet = quiet
        self.seed = 555

        if not self.quiet:
            print(f"**** Loading data_root: {data_root} ****") 
            print(f" shuffle seed:{self.seed}")

        self.__recurse_images(self=self, recurse_root=data_root)
        
        import random
        random.Random(self.seed).shuffle(self.image_paths)        
        self.seed = self.seed + 1 
        
        print(f"**** Done loading. Loaded {len(self.image_paths)} images from data_root: {self.data_root} ****") if not self.quiet else None

    def get_all_images(self):
        return self.image_paths

    @staticmethod
    def __recurse_images(self, recurse_root):
        i = 0
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            if os.path.isfile(current):
                i += 1
                self.image_paths.append(current)

        print(f"  ** Found {str(i).rjust(5,' ')} files in", recurse_root) if not self.quiet else None

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_images(self=self, recurse_root=dir)