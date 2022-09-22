import torch.nn as nn
import torch

class modelArch(nn.Module):
    def __init__(self, pretrained_model) -> None:
        super(modelArch, self).__init__()
        self.pretrained = pretrained_model
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.pretrained.fc.in_features # Fully connected layer input features
        # another layer
        self.pretrained.fc = nn.Linear(num_ftrs, 2) # Why is it Linear?
        
    def setChildren(self, stop, freeze):
        for idx, child in enumerate(self.pretrained.children()):
            if idx < stop:
                for param in child.parameters():
                    param.requires_grad = freeze
            else:
                break
    
    def forward(self, input):
        #pass the inputs to the model  
        preds = self.pretrained(input)
        return preds
