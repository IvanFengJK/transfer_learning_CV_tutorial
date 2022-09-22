from data.dataset import Dataset
from experiment import Experiment
from config.config import cfg
from torchvision import models
import torch
from models.tl_arch import modelArch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

### Loading Data Phase ###
data = Dataset(cfg)
# data.sampleImages()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Training by finetuning the convnet phase ###
# Load pretrained model
pretrained = models.resnet18(pretrained=True)
model = modelArch(pretrained)
model.setALL(False)
model.setFC(True)

experiment = Experiment(model,
                        data, 
                        device, 
                        num_epochs=cfg.train.num_epochs)

# Function to calculate loss
criterion = nn.CrossEntropyLoss() 

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=cfg.model_ft.optimizer.lr, momentum=cfg.model_ft.optimizer.momentum)

# Decay Learning rate by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model_ft.lr_scheduler.step_size, gamma=cfg.model_ft.lr_scheduler.gamma)

experiment.setTrainer(criterion, optimizer, scheduler)
experiment.train_model()
experiment.visualize_model(cfg.experiment.num_sample_visual)
experiment.displayAccHist()