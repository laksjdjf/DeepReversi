from torch.utils.data import Dataset
import torch
import numpy as np

class BoardDataset:
    def __init__(self, path) -> None:
        data = np.load(path)
        self.features = data["features"].astype(np.int8)
        self.results = data["results"]
        self.moves = np.eye(64)[data["moves"]]
        self.evals = data["evals"]
    def __getitem__(self,index: int):
        return torch.tensor(self.features[index],dtype=torch.float),torch.tensor(self.moves[index],dtype=torch.float),torch.tensor(self.results[index],dtype=torch.float),torch.tensor(self.evals[index],dtype=torch.float)
    def __len__(self):
        return len(self.results)
