from torchvision import datasets, transforms
import torchvision
from imshow import imshow
import torch
import os

class Dataset():
    def __init__(self, cfg) -> None:
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path =os.path.join(project_dir, cfg.dataset.script_path)
        self.data_dir = path
        self.setImageDataset()
        self.setDataLoader(cfg.dataset.batch_size)
        self.setDatasetSize()
        self.setClassName()
        print("Data is loaded and there are {} classes: ".format(len(self.class_names)), end="")
        for i in self.class_names:
            print(i, end=" ")
        print("")

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

    def sampleImages(self, writer):
        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        
        fig = imshow(out, title=[self.class_names[x] for x in classes], show=True)
        writer.add_figure('Sample Input Images', fig)
