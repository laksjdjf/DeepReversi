from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
import random

class BoardDataset:
    def __init__(self, path) -> None:
        data = np.load(path)
        self.features = data["features"].astype(np.int8)
        self.results = data["results"]
        self.moves = np.eye(64)[data["moves"]].astype(np.int8)
        self.evals = data["evals"]
        self.rotate = transforms.RandomRotation((90,90))
        self.transform = transforms.Compose(
            [
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
             ]
        )
    def __getitem__(self,index: int):
        feature = self.features[index]
        move = self.moves[index].reshape(1,8,8)
        board = torch.tensor(np.concatenate([feature,move]))
        if random.random() < 0.5:
            board = self.rotate(board)
        feature_tensor, move_tensor = self.transform(board).split([2,1],dim=0)
        move_tensor = move_tensor.reshape(64)
        
        return feature_tensor.float(),move_tensor.float(),torch.tensor(self.results[index],dtype=torch.float),torch.tensor(self.evals[index],dtype=torch.float)
    def __len__(self):
        return len(self.results)
