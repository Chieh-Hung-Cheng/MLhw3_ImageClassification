import os
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from Config import Config
import matplotlib.pyplot as plt


augmented_transformer = transforms.Compose([
    transforms.Resize((168, 168), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=60)], p=0.5),
    transforms.CenterCrop((128,128))
])

simple_transformer = transforms.Compose([transforms.Resize((128,128), antialias=True)])


class ImageDataset(Dataset):
    def __init__(self, split_name):
        self.split_name = split_name
        self.split_path = os.path.join(Config.data_path, split_name)
        self.filenames = sorted(os.listdir(self.split_path))
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.filenames[item]
        # Image
        img = cv2.cvtColor(cv2.imread(os.path.join(self.split_path, filename)), cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img).permute(2, 0, 1)
        if self.split_name=="train" or self.split_name=="valid":
            img = augmented_transformer(img)
            label = int(filename.removesuffix(".jpg").split("_")[0])
            return img, label
        elif self.split_name == "test":
            id = filename.removesuffix(".jpg")
            img = simple_transformer(img)
            return img, id
        else:
            raise ValueError("split type not found")

def show_tensor_image(tensor_image):
    plt.imshow(tensor_image.permute(1,2,0).to(torch.int32))
    plt.show()

def given_path_test():
    img = cv2.cvtColor(cv2.imread(r"G:\My Drive\Chronical\2023Spring\ML_drive\MLHW3\data\train\0_83.jpg"),
                       cv2.COLOR_BGR2RGB)
    img = torch.FloatTensor(img).permute(2, 0, 1)
    show_tensor_image(img)

if __name__ == "__main__":
    Config.data_path = os.path.join(os.getcwd(), "data")
    image_dataset = ImageDataset("train")
    for _ in range(10):
        show_tensor_image(image_dataset[4][0])
