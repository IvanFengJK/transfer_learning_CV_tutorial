# transfer_learning_CV_tutorial

This is a practice to organise my files for machine learning and the code are taken from https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/74249e7f9f1f398f57ccd094a4f3021b/transfer_learning_tutorial.ipynb#scrollTo=XNS7leCT41Tu

Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    |
    ├── build                   <- Folder that contains files for building the environment 
    │   ├── docker-compose.yml  <- docker-compose file for quickly building containers
    │   ├── Makefile            <- Makefile which will be ran when building the docker image
    │   └── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                           generated with `pip freeze > requirements.txt`
    │── Dockerfile              <- Dockerfile for building docker image (unfortunately it has to be in root for it to work)
    |
    ├── data                    <- Download data from clearml here
    |   ├── train        
    │   └── valid
    |     
    ├── models                  <- Download/load pretrained model/save trained model locally here
    |   ├── vgg        
    │   └── elmo
    |   └── trained_models      <- Folder that contains the trained model weights
    │       └── model_weights.ckpt     
    |
    ├── src                     <- Source code for use in this project.
    │   │
    │   ├── main.py             <- Code to run for task initialization,  sending to remote, download datasets, starting experimentation
    |   |
    │   ├── experiment.py       <- Experimentation defining the datasets, trainer, epoch behaviour and running training
    |   |
    │   ├── config
    |   │   ├── config.py       <- Boilerplate code for config loading.yaml   
    |   │   └── config.yaml     <- Configfile for parameters
    |   |
    │   ├── data                <- Scripts related to data procesing
    │   │   ├── dataset.py
    │   │   ├── postprocessing.py
    │   │   ├── preprocessing.py
    │   |   ├── transforms.py
    |   |   └── common          <- common reusable transformation modules
    |   │       └── transforms.py 
    │   │
    |   ├── model               <- Scripts related to module architecture
    |   │   ├── model.py        <- Main model file chaining together modules 
    |   │   └── modules         <- Folder containing model modules
    |   |       ├── common
    |   |       |   └── crf.py 
    |   |       ├── encoder.py           
    |   |       └── decoder.py           
    │   │
    │   └── evaluation          <- Scripts to generate evaluations of model e.g. confusion matrix etc.
    |       ├── visualize.py           
    |       └── common 
    |           └── metrics.py
    |
    ├── tests                   <- Folder where all the unit-tests are
    |
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                           the creator's initials, and a short `-` delimited description, e.g.
    │                           `1.0-jqp-initial-data-exploration`.
    |
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    |
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

To run project, you need to have the docker extension:
```bash
$ cd build
$ docker-compose up
```

Attach the shell from container tl:
```bash
$ python3 main.py
```

Attach the shell from container tb:
```bash
$ tensorboard --bind_all --logdir=runs
```

Open your browser and type `localhost:6006`. This might take a while to load but you will see the training graph now