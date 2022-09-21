from data.dataset import Dataset
from experiment import Experiment
from models.models import Models
from config.config import cfg
import torch
import os


### Loading Data Phase ###

dataset_path =os.path.join(os.path.dirname(os.path.abspath(__file__)),cfg.dataset.script_path)
data = Dataset(cfg.dataset.script_path, cfg.dataset.batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Training by finetuning the convnet phase ###
# Load pretrained model
model = Models(device, len(data.class_names))
model.setModelConv()

experiment = Experiment( model,
                            data, 
                            device, 
                            num_epochs=cfg.train.num_epochs)

experiment.train_model()
experiment.visualize_model(cfg.experiment.num_sample_visual)
experiment.displayAccHist()