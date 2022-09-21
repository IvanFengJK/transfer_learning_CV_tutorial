from data.transforms import get_transforms
from imshow import imshow
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from experiment import train_model
from experiment import visualize_model
from evaluation.visualize import visualiseList
from evaluation.common.metric import evaluation


### Loading Data Phase ###
print(torch.__version__)
data_dir = "/../data"
# Set batch size to 4
data = get_transforms(data_dir, 4)

# Get a batch of training data
inputs, classes = next(iter(data.dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
# Display a batch of pictures
# imshow(out, title=[data.class_names[x] for x in classes], show=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""### Training by finetuning the convnet phase ###
# Load pretrained model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features # Fully connected layer input features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device) # Attach to GPU

criterion = nn.CrossEntropyLoss() # Function to calculate loss

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay Learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft, train_acc_hist, val_acc_hist = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data.dataloaders, device, data.dataset_sizes, num_epochs=200)

#visualize_model(model_ft, data.dataloaders, device, data.class_names, show=False)

visualiseList(train_acc_hist, val_acc_hist)"""

### Evaluation phase ###
# evaluation(model_ft, data.dataloaders, device)

### Training by using ConvNet as fixed feature extractor ###

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False #Fix the features

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features # ully connected layer input features
model_conv.fc = nn.Linear(num_ftrs, 2) # Why is it Linear?

model_conv = model_conv.to(device) # Attach to GPU

criterion = nn.CrossEntropyLoss() # Function to calculate loss

# Observe that only parameters of final layer are being optimized as
# opposed to before.
# Stochastic gradient descent with learning rae 0.001 ad momentum of 0.9
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv, train_acc_hist, val_acc_hist = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, data.dataloaders, device, data.dataset_sizes, num_epochs=100)

visualiseList(train_acc_hist, val_acc_hist)

### Evaluation phase ###
#evaluation(model_conv, data.dataloaders, device)
