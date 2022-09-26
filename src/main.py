import datetime
from data.dataset import Dataset
from experiment import Experiment
from config.config import cfg
from torchvision import models
import torchvision
import torch
from models.tl_arch import modelArch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import PIL.Image
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    ### Loading Data Phase ###
    data = Dataset(cfg)
    writer = SummaryWriter('runs/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data.sampleImages(writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Training by finetuning the convnet phase ###
    # Load pretrained model
    pretrained = models.resnet18(pretrained=True)
    model = modelArch(pretrained)
    model.setChildren(cfg.model.num_children, False)
    model.to(device)

    images, labels = next(iter(data.dataloaders['train']))
    writer.add_graph(model, images.to(device))

    experiment = Experiment(model,
                            data, 
                            device, 
                            num_epochs=cfg.train.num_epochs)

    # Function to calculate loss
    criterion = nn.CrossEntropyLoss() 
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=cfg.model.optimizer.lr, momentum=cfg.model.optimizer.momentum)
    # Decay Learning rate by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_scheduler.step_size, gamma=cfg.model.lr_scheduler.gamma)

    experiment.setTrainer(criterion, optimizer, scheduler)
    experiment.train_model(cfg.experiment.patience, writer)
    experiment.visualize_model(cfg.experiment.num_sample_visual, writer)
    writer.close()