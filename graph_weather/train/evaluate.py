import torch
import time
import xarray as xr
import numpy as np
from tqdm import tqdm
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss
import pdb
import os
import json
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='../../config', config_name='config')
def main(cfg: DictConfig``):

	vars = ['z', 't']

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	z = xr.open_mfdataset(f'{cfg.data_dir}/geopotential/*.nc', combine='by_coords')
	t = xr.open_mfdataset(f'{cfg.data_dir}/temperature/*.nc', combine='by_coords')
	ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.
		
	ds_train = ds.sel(time=slice(cfg.train_start, cfg.train_end))
	ds_test = ds.sel(time=slice(cfg.test_start, cfg.test_end))

	lat_lons = [(lat, lon) for lat in ds_train.lat.values for lon in ds_train.lon.values]
	model = GraphWeatherForecaster(lat_lons, 
		edge_dim=cfg.model_dim,
		hidden_dim_processor_edge=cfg.model_dim,
		node_dim=cfg.model_dim,
		hidden_dim_processor_node=cfg.model_dim,
		hidden_dim_decoder=cfg.model_dim,
		feature_dim=2,
		aux_dim=0,
		num_blocks=cfg.num_blocks,
		resolution=cfg.resolution,
	).to(device)


if __name__ == '__main__':
	main()