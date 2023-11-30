import os, math, logging
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from livelossplot import PlotLosses

from .utils import sample, is_notebook
from .dataset import LayoutDataset

logger = logging.getLogger(__name__)

class LayoutTransformerTrainer:
    def __init__(self, config, workdir: Optional[str]=None):
        self.config = config
        if workdir is not None:
            self.workdir = Path(workdir)
            if self.workdir.exists() == False:
                raise Exception(f"{workdir} does not exist!")
        else:
            self.workdir = Path('')
                # take over whatever gpus are on the system
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self, device):
        if is_notebook():
            liveplot = PlotLosses()
        else:
            liveplot = None

        train_dataset = LayoutDataset(self.config.dataset_path+'/train.json',
                                      config=self.config.dataset)
        val_dataset = LayoutDataset(self.config.dataset_path+'/val.json',
                                      config=self.config.dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                      shuffle=self.config.train_shuffle,
                                      drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.eval_batch_size,
                                    shuffle=self.config.train_shuffle,
                                    drop_last=False)
        
        


        
        
