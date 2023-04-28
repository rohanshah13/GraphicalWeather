import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from configargparse import ArgParser
from pytorchtools import EarlyStopping
import json
import random

from matplotlib import pyplot as plt

class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        print('Lead time = ', lead_time)

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))
            except KeyError:
                data.append(ds[var])
        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.data = self.data.transpose('time', 'level', 'lat', 'lon')
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.ceil(self.n_samples / self.batch_size))
        return self.n_samples

    def __getitem__(self, i):
        'Generate one batch of data'
        # idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        idxs = self.idxs[i:i+1]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        return X[0], y[0]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)

def get_data(config):
    z = xr.open_mfdataset(f'{config.dataset_path}/geopotential_500/*.nc', combine='by_coords')
    t = xr.open_mfdataset(f'{config.dataset_path}/temperature_850/*.nc', combine='by_coords')
    ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

    # TODO: Flexible valid split
    ds_train = ds.sel(time=slice(*config.train_years))
    ds_valid = ds.sel(time=slice(*config.valid_years))
    ds_test = ds.sel(time=slice(*config.test_years))

    dic = {var: None for var in config.vars}
    dg_train = DataGenerator(ds_train, dic, config.lead_time, batch_size=config.batch_size)
    dg_valid = DataGenerator(ds_valid, dic, config.lead_time, batch_size=config.batch_size, mean=dg_train.mean,
                             std=dg_train.std, shuffle=False)
    dg_test = DataGenerator(ds_test, dic, config.lead_time, batch_size=config.batch_size, mean=dg_train.mean,
                            std=dg_train.std, shuffle=False)
    # print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')
    training_loader = torch.utils.data.DataLoader(dg_train, batch_size=config.batch_size)
    validation_loader = torch.utils.data.DataLoader(dg_valid, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(dg_test, batch_size=config.batch_size)
    return training_loader, validation_loader, test_loader

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.savefig('results/plot.png')
    plt.show()

def plot_images_f(images, path):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.savefig(f'results/imgs/{path}.png')
    plt.show()

def plot_images_weatherbench(gt, images, path, idx):
    plt.subplot(1, 4, 1)
    plt.imshow(gt[0, :, :].cpu())
    plt.subplot(1, 4, 2)
    plt.imshow(gt[1, :, :].cpu())
    plt.subplot(1, 4, 3)
    plt.imshow(images[0, :, :].cpu())
    plt.subplot(1, 4, 4)
    plt.imshow(images[1, :, :].cpu())
    if not os.path.exists(f'/home/scratch/vdas/weatherbench/results/{path}'):
        os.makedirs(f'/home/scratch/vdas/weatherbench/results/{path}')
    plt.savefig(f'/home/scratch/vdas/weatherbench/results/{path}/{idx}.png')



def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False