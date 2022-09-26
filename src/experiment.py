import copy
import time
import torch
import matplotlib.pyplot as plt
from imshow import imshow
from tqdm import tqdm
from evaluation.visualize import visualiseList
from evaluation.common.metric import evaluation
import torchvision

class Experiment():
    def __init__(self, model, data, device, num_epochs=25) -> None:
        self.model = model
        self.dataloaders = data.dataloaders
        self.device = device
        self.dataset_sizes = data.dataset_sizes
        self.class_names = data.class_names
        self.num_epochs = num_epochs

    def setTrainer(self, criterion, optimizer, lr_scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        
    def train_model(self, patience, writer):
        since = time.time()
        # Make a copy of the model instead of a reference
        best_model_wts = copy.deepcopy(self.model.state_dict())
        self.trained_model = copy.deepcopy(self.model)
        best_acc = 0.0
        train_acc_hist = []
        val_acc_hist = []
        trigger = 0
        for epoch in tqdm(range(self.num_epochs)):

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(inputs)
                        # Calculate loss function base on criterion
                        _, preds = torch.max(output, 1)
                        loss = self.criterion(output, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                writer.add_scalars('loss', {phase: round(epoch_loss,4)}, epoch)
                writer.add_scalars('accuracy', {phase: round(epoch_acc.item(),4)}, epoch)

                
                # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # My own early stopping
                if phase == 'val' and len(val_acc_hist) > 1:
                    if val_acc_hist[-2] > val_acc_hist[-1]:
                        trigger += 1
                    else:
                        trigger = 0
            if trigger >= patience:
                print(trigger)
                print("Early Stopping triggered")
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.train_acc_hist= train_acc_hist
        self.val_acc_hist = val_acc_hist


    def visualize_model(self, num_images, writer):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    fig = imshow(inputs.cpu().data[j], f'predicted: {self.class_names[preds[j]]}')
                    writer.add_figure(f'Sample Output Image {j}', fig)
                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)

    def displayAccHist(self):
        visualiseList(self.train_acc_hist, self.val_acc_hist)

    def displayEvaReport(self):
        evaluation(self.model, self.dataloaders, self.device)
