import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)


class StairDataset(Dataset):
    def __init__(self, image_dir, image_size, random_seed=42):
        self.name = 'StairDataset'
        self.image_dir = image_dir
        self.image_size = image_size
        self.random_seed = random_seed
    
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.common_transform = transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.CenterCrop(size=self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def dataset_split(self, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
        assert abs(1 - (train_ratio + valid_ratio + test_ratio)) < 1e-6
        
        dataset = ImageFolder(self.image_dir)
        full_size = len(dataset)
        
        train_indices, test_indices  = train_test_split(list(range(full_size)), test_size=test_ratio, random_state=self.random_seed)
        train_indices, valid_indices = train_test_split(train_indices, test_size=valid_ratio / (train_ratio + valid_ratio), random_state=self.random_seed)

        train_dataset = SubsetDataset(dataset, train_indices, transform=self.train_transform)
        valid_dataset = SubsetDataset(dataset, valid_indices, transform=self.common_transform)
        test_dataset  = SubsetDataset(dataset, test_indices,  transform=self.common_transform)
        
        return train_dataset, valid_dataset, test_dataset

    def dataset_generate(self):
        return ImageFolder(self.image_dir, self.common_transform)
