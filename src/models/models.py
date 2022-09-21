from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from config.config import cfg

class Models():
    def __init__(self, device, class_size):
        self.device = device
        self.class_size = class_size
        self.model = models.resnet18(pretrained=True)
    
    def setModelFT(self):
        ### Training by finetuning the convnet phase ###
        # Load pretrained model
        self.num_ftrs = self.model.fc.in_features # Fully connected layer input features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model.fc = nn.Linear(self.num_ftrs, self.class_size)

        self.model = self.model.to(self.device) # Attach to GPU

        self.criterion = nn.CrossEntropyLoss() # Function to calculate loss

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.model_ft.optimizer.lr, momentum=cfg.model_ft.optimizer.momentum)

        # Decay Learning rate by a factor of 0.1 every 7 epochs
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.model_ft.lr_scheduler.step_size, gamma=cfg.model_ft.lr_scheduler.gamma)
    
    def setModelConv(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            #print (name)

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features # ully connected layer input features
        self.model.fc = nn.Linear(num_ftrs, 2) # Why is it Linear?

        model_conv = self.model.to(self.device) # Attach to GPU

        self.criterion = nn.CrossEntropyLoss() # Function to calculate loss

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        # Stochastic gradient descent with learning rae 0.001 ad momentum of 0.9
        self.optimizer = optim.SGD(model_conv.fc.parameters(), lr=cfg.model_ft.optimizer.lr, momentum=cfg.model_ft.optimizer.momentum)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=cfg.model_ft.lr_scheduler.step_size, gamma=cfg.model_ft.lr_scheduler.gamma)
