import os, math, logging
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from livelossplot import PlotLosses

from .utils import sample, is_notebook
from .model import GPTConfig, GPT
from .dataset import LayoutDataset
import wandb
import time

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
        self.model = self._create_model(config)
        wandb.init(project='LayoutTransformer', config=config, 
                   name=str(int(time.time()))+'_'+config.name)
        self.device = 'cpu'
        # take over whatever gpus are on the system
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        if is_notebook():
            liveplot = PlotLosses()
        else:
            liveplot = None

        self.train_dataset = LayoutDataset(self.config.dataset_path+'/train.json',
                                      config=self.config.dataset)
        self.val_dataset = LayoutDataset(self.config.dataset_path+'/val.json',
                                      config=self.config.dataset)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                      shuffle=self.config.train_shuffle,
                                      drop_last=False)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.eval_batch_size,
                                    shuffle=self.config.train_shuffle,
                                    drop_last=False)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = raw_model.configure_optimizers(self.config.optimizer)
        pad_token = self.train_dataset.pad_token

        if self.config.ckpt_path is not None:
            ckpt = self._load_ckpt(self.config.ckpt_path)
            
            self.model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])

            metric_history = ckpt['metric_history']
            min_val_loss = ckpt['min_loss']
            start_epoch = ckpt['epoch']
            self.iters = ckpt['iters']

            print("Loaded epoch {} - Loss: {:.6f}".format(start_epoch, min_val_loss))
        else:
            metric_history = {'loss': [],
                            'val_loss': []}    
            min_val_loss = float('inf')
            start_epoch = 0
            self.iters = 0
            print("Start training from scratch")

        for epoch in range(start_epoch+1, self.config.epoch+1):
            # Running on train and validation set
            train_loss = self._epoch(epoch, 'train', train_dataloader, optimizer, pad_token)
            val_loss = self._epoch(epoch, 'val', val_dataloader, optimizer, pad_token)
            
            # Metrics
            metric_history['loss'].append(train_loss)
            metric_history['val_loss'].append(val_loss)
            # Log messages
            logs = {}
            logs["loss"] = metric_history["loss"][-1]
            logs["val_loss"] = metric_history["val_loss"][-1]
            if liveplot:
                liveplot.update(logs)
                liveplot.send()
            else:
                print('Epoch {} train/val loss: {:.6f} / {:.6f}'.format(epoch,
                                                                    logs['loss'],
                                                                    logs['val_loss']))
            
            # Checkpoint
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                self._save_ckpt({
                    'epoch': epoch,
                    'metric_history': metric_history,
                    'min_loss': min_val_loss,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iters': self.iters
                }, 'best.pth.tar')
            if (epoch % self.config.save_every_epoch) == 0:
                self._sample_layout(epoch)
                self._save_ckpt({
                    'epoch': epoch,
                    'metric_history': metric_history,
                    'min_loss': min_val_loss,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iters': self.iters
                }, 'epoch{}.pth.tar'.format(epoch))
            
                
    def _create_model(self, config):
        #TODO - change 256 into config
        vocab_size = 256 + config.dataset.NUMBER_LABELS + 3 # bos, eos and pad tokens
        block_size = (config.limit * 5) + 2 # for bos and eos tokens
        mconfig = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                            n_layer=config.num_layers, n_head=config.num_heads,
                            n_emb=config.qkv_dim)
        return GPT(mconfig)
    
    def _epoch(self, epoch, split, dataloader, opt, pad_token):
        is_train = split == 'train' # True or False based on split
        model = self.model
        model.train(is_train)
        n_batches = len(dataloader)
        epoch_loss = 0
        for x,y in dataloader:
            x = x.to(self.device)
            if not is_train:
                self.fixed_x = x[:min(4, len(x))]
            y = y.to(self.device)

            with torch.set_grad_enabled(is_train):
                logits, loss = self.model(x,y, pad_token=pad_token)
                loss = loss.mean()
                epoch_loss += loss
            
            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                               self.config.optimizer.grad_norm_clip)
                opt.step()
                self.iters += 1
                if self.config.optimizer.lr_decay:
                    if self.iters < self.config.optimizer.warmup_iters:
                        lr_mult = float(self.iters) / float(max(1, self.config.optimizer.warmup_iters)) # Linear Ramp
                    else:
                        progress = float(self.iters - self.config.optimizer.warmup_iters) / \
                                    float(max(1, self.config.optimizer.final_iters - self.config.optimizer.warmup_iters))
                        lr_mult = max(0.1, 0.5*(1.0 + math.cos(math.pi * progress))) 
                    lr = self.config.optimizer.lr * lr_mult
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.optimizer.lr
                # For training, report loss every batch
                wandb.log({
                'batch_train_loss': loss,
                'lr': lr,
                'epoch': epoch
                }, step=self.iters)
            del x, y, logits
        epoch_loss = (epoch_loss / n_batches).item()
        # For Validation, only report the epoch loss    
        if not is_train:
            wandb.log({'epoch_val_loss': epoch_loss}, step=self.iters)
        return epoch_loss

    def _sample_layout(self, epoch):
        # Based Layout
        layouts = self.fixed_x.detach().cpu().numpy()
        input_layouts = [self.train_dataset.render(layout) for layout in layouts]
        # Reconstruction Layout
        x_cond = self.fixed_x.to(self.device)
        logits, _ = self.model(x_cond)
        probs = F.softmax(logits, dim=-1)
        _, y = torch.topk(probs, k=1, dim=-1)
        layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
        recon_layouts = [self.train_dataset.render(layout) for layout in layouts]

        # Samples - Random
        layouts = sample(self.model, x_cond[:, :6], steps=self.train_dataset.seq_len,
                            temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
        
        # samples - deterministic
        layouts = sample(self.model, x_cond[:, :6], steps=self.train_dataset.seq_len,
                            temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
        sample_det_layouts = [self.train_dataset.render(layout) for layout in layouts]

        wandb.log({
            "input_layouts": [wandb.Image(pil, caption='input_{:02d}_{:02d}.png'.format(epoch,i))
                                for i, pil in enumerate(input_layouts)],
            "recon_layouts": [wandb.Image(pil, caption='recon_{:02d}_{:02d}.png'.format(epoch,i))
                                for i, pil in enumerate(recon_layouts)],
            "sample_random_layouts": [wandb.Image(pil, caption='sample_random_{:02d}_{:02d}.png'.format(epoch,i))
                                        for i, pil in enumerate(sample_random_layouts)],
            "sample_det_layouts": [wandb.Image(pil, caption='sample_det_{:02d}_{:02d}.png'.format(epoch,i))
                                    for i, pil in enumerate(sample_det_layouts)],
        }, step=self.iters)

    def _save_ckpt(self, state, filename):
        out_path = self.workdir/filename
        torch.save(state, out_path)
    
    def _load_ckpt(self, path):
        state = torch.load(path, map_location=self.device)
        return state