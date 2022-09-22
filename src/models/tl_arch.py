import torch.nn as nn
import torch

class modelArch(nn.Module):
    def __init__(self, pretrained_model) -> None:
        super(modelArch, self).__init__()
        self.pretrained = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]))

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = pretrained_model.fc.in_features # Fully connected layer input features
        self.fc = nn.Linear(num_ftrs, 2) # Why is it Linear?
        
    def setALL(self, freeze):
        for param in self.parameters():
            param.requires_grad = freeze

    def setPretrained(self, freeze):
        for param in self.pretrained.parameters():
            param.requires_grad = freeze

    def setFC(self, freeze):
        for param in self.fc.parameters():
            param.requires_grad = freeze

    def setLayerBlock(self, layer_num, block_num):
        for param in getattr(getattr(self, 'layer'+str(layer_num)),str(block_num)).parameters():
            param.requires_grad = True

    def unfreezeLayer(self, layer_num):
        for param in getattr(self, 'layer'+str(layer_num)).parameters():
            param.requires_grad = True
    
    def forward(self, inputs):
        outputs = self(inputs)
        # Take the max of the prediction
        preds = torch.max(outputs, 1)[1]
        return preds
