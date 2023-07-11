import os
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Config import Config


class ImageParser:
    def __init__(self, split_type: str):
        self.split_path = os.path.join(Config.data_path, split_type)
        self.split_type = split_type
        self.image_list = []
        self.label_list = []
        self.id_list =[]
        self.parse_images()

    def parse_images(self):
        for filename in tqdm(os.listdir(self.split_path)):
            filename_no_extension = filename.strip(".jpg")
            # Label
            label, id = [i for i in filename_no_extension.split("_")]
            self.label_list.append(int(label))
            self.id_list.append(id)
            original_image = cv2.imread(os.path.join(self.split_path, filename))
            # Image array
            resize_shape = (128, 128)
            resized_image = cv2.resize(original_image, resize_shape)
            resized_image = torch.permute(torch.FloatTensor(resized_image), (2,0,1))
            self.image_list.append(resized_image)


class ImageDataset(Dataset):
    def __init__(self, feature_list, label_list):
        self.tensor_list = feature_list

        self.label_list = None
        if label_list is not None:
            self.label_list = torch.LongTensor(label_list)

    def __len__(self):
        if self.label_list is not None:
            assert len(self.tensor_list) == len(self.label_list)
        return len(self.tensor_list)

    def __getitem__(self, item):
        if self.label_list is not None:
            return self.tensor_list[item], self.label_list[item]
        else:
            return self.tensor_list[item]


if __name__ == "__main__":
    pass