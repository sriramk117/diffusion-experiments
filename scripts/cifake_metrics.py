import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class CIFAKEDatasetParser(Dataset):

    def __init__(self, path, split, resolution, size): 
        # Check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")
        # Check if split is valid
        if split not in ["train", "test"]:
            raise ValueError(f"Split {split} is not valid.")
        # Check if resolution is valid
        if resolution not in [32, 64, 128]:
            raise ValueError(f"Resolution {resolution} is not valid.")
        
        # Set attributes
        self.path = path
        self.real_data = []
        self.fake_data = []
        self.split = split
        self.resolution = resolution
        self.size = size
        self.parse()

    def file_is_image(self, file):
        # Check if file is an image
        return file.lower().endswith(("jpg", "png", "jpeg"))

    def parse(self):
        # Load data
        path = os.path.join(self.path, self.split)

        # Randomly extract some self.size number of real and fake images from the dataset
        random.seed(42)

        self.real_data = [os.path.join(path, "REAL", img) 
                          for img in random.sample(os.listdir(os.path.join(path, "REAL")), self.size) if self.file_is_image(img)]
        self.fake_data = [os.path.join(path, "FAKE", img)
                            for img in random.sample(os.listdir(os.path.join(path, "FAKE")), self.size) if self.file_is_image(img)]

    def process_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        transform = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.real_data) + len(self.fake_data)
    
    def __getitem__(self, idx):
        if idx < len(self.real_data):
            img = self.process_image(self.real_data[idx])
            return img
        else:
            img = self.process_image(self.fake_data[idx - len(self.real_data)])
            return img
        


    


