from torchvision import datasets, transforms
import torch
import os

class Data():
    def __init__(self, path:str) -> None:
        self.data_dir = path

    def setImageDataset(self) -> None:
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.image_datasets = {x:   datasets.ImageFolder(   os.path.join(self.data_dir, x),
                                                            self.data_transforms[x])
                                    for x in list(self.data_transforms.keys())}

    def setDataLoader(self, n) -> None:
        self.dataloaders = {x:  torch.utils.data.DataLoader(self.image_datasets[x],      
                                                            batch_size=n,
                                                            shuffle=True, 
                                                            num_workers=4)
                                for x in list(self.image_datasets.keys())}

    def setDatasetSize(self) -> None:
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in list(self.image_datasets.keys())}

    def setClassName(self) -> None:
        self.class_names = self.image_datasets['train'].classes

def get_transforms(path:str, batchsize=4) -> torch.utils.data.DataLoader:
    pwd = os.getcwd()
    data = Data(pwd+path)
    data.setImageDataset()
    data.setDataLoader(batchsize)
    data.setDatasetSize()
    data.setClassName()
    print("Data is loaded and there are {} classes: ".format(len(data.class_names)), end="")
    for i in data.class_names:
        print(i, end=" ")
    print("")
    return data