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




def load_test_data(path, var, years):
    """
    Load the test dataset. If z return z500, if t return t850.
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window

    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    # if var in ['z', 't']:
    #     if len(ds["level"].dims) > 0:
    #         try:
    #             ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
    #         except ValueError:
    #             ds = ds.drop('level')
    #     else:
    #         assert ds["level"].values == 500 if var == 'z' else ds["level"].values == 850
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse

class MyDataset(torch.utils.data.IterableDataset):
	def __init__(self):
		self.input_data = torch.randn((200, 64800, 2))
		self.output_data = torch.randn((200, 64800, 2))

	def __iter__(self):
		for i in range(200):
			yield self.input_data[i], self.output_data[i]

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

		data = []
		generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
		for var, levels in var_dict.items():
			try:
				data.append(ds[var].sel(level=levels))
			except ValueError:
				data.append(ds[var].expand_dims({'level': generic_level}, 1))
			except KeyError:
				data.append(ds[var])
		# ipdb.set_trace()
		self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
		# import pdb;pdb.set_trace()
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
		# return 16
		return self.n_samples

	def __getitem__(self, i):
		'Generate one batch of data'
		# idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
		idxs = self.idxs[i:i+1]
		X = self.data.isel(time=idxs).values
		y = self.data.isel(time=idxs + self.lead_time).values
		# print(X.shape)
		# print(y.shape)
		return X[0], y[0]

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.idxs = np.arange(self.n_samples)
		if self.shuffle == True:
			np.random.shuffle(self.idxs)

def create_iterative_forecast_rmse(model, dataloader, dg, device, valid):
	# model: model to predict at the 6h lead time
	# dataloader: dl with data (x, y) where y = x + LD (LD is a multiple of 6 and can be found from dg.lead_time)
	# algo: with the dataloade
	preds = None
	labels_acc = None
	model.eval()
	with torch.no_grad():
		for i, data in tqdm(enumerate(dataloader)):		
			inputs, labels = process_data(data, device)
			for forecast_num in range(dg.lead_time // 6):
				inputs = model(inputs)
			outputs = inputs
			# import pdb;pdb.set_trace()
			outputs = outputs.cpu().detach().numpy()
			labels = labels.cpu().detach().numpy()
			if preds is None:
				preds = outputs
			else:
				preds = np.concatenate((preds, outputs), axis=0)
			if labels_acc is None:
				labels_acc = labels
			else:
				labels_acc = np.concatenate((labels_acc, labels), axis=0)

	preds = preds.reshape(-1, 64, 128, 2)
	preds = preds.transpose(0, 3, 1, 2)
	labels_acc = labels_acc.reshape(-1, 64, 128, 2)
	labels_acc = labels_acc.transpose(0, 3, 1, 2)
	preds = preds*np.array(dg.std.values).reshape((1,2,1,1)) + np.array(dg.mean.values).reshape((1,2,1,1))
	labels_acc = labels_acc*np.array(dg.std.values).reshape((1,2,1,1)) + np.array(dg.mean.values).reshape((1,2,1,1))
	# pdb.set_trace()
	labels = valid.load().sel(time=dg.valid_time)
	lat = np.reshape(labels.lat.values, (1, -1, 1))
	weights_lat = np.cos(np.deg2rad(lat))
	weights_lat /= weights_lat.mean()
	lev_idx = 0
	rmse_dict = {}
	for var, levels in dg.var_dict.items():
		preds_var = preds[:, lev_idx]
		# add for labels_var (idx = lev_idx)
		# labels_var = np.reshape(labels_acc[:, :, lev_idx], (-1, 64, 128))
		labels_var = labels_acc[:, lev_idx]
		# labels_var = getattr(labels, var).values
		lev_idx += 1
		error = preds_var - labels_var
		rmse = np.sqrt(((error)**2 * weights_lat).mean())
		rmse_dict[var] = rmse
	return rmse_dict

def create_predictions_rmse(model, dataloader, dg, device, valid):
	preds = None
	labels_acc = None
	model.eval()
	with torch.no_grad():
		for i, data in tqdm(enumerate(dataloader)):		
			inputs, labels = process_data(data, device)
			outputs = model(inputs)
			outputs = outputs.cpu().detach().numpy()
			labels = labels.cpu().detach().numpy()
			if preds is None:
				preds = outputs
			else:
				preds = np.concatenate((preds, outputs), axis=0)
			if labels_acc is None:
				labels_acc = labels
			else:
				labels_acc = np.concatenate((labels_acc, labels), axis=0)

	preds = preds.reshape(-1, 64, 128, 2)
	preds = preds.transpose(0, 3, 1, 2)
	labels_acc = labels_acc.reshape(-1, 64, 128, 2)
	labels_acc = labels_acc.transpose(0, 3, 1, 2)
	preds = preds*np.array(dg.std.values).reshape((1,2,1,1)) + np.array(dg.mean.values).reshape((1,2,1,1))
	labels_acc = labels_acc*np.array(dg.std.values).reshape((1,2,1,1)) + np.array(dg.mean.values).reshape((1,2,1,1))
	# pdb.set_trace()
	labels = valid.load().sel(time=dg.valid_time)
	lat = np.reshape(labels.lat.values, (1, -1, 1))
	weights_lat = np.cos(np.deg2rad(lat))
	weights_lat /= weights_lat.mean()
	lev_idx = 0
	rmse_dict = {}
	for var, levels in dg.var_dict.items():
		preds_var = preds[:, lev_idx]
		# add for labels_var (idx = lev_idx)
		# labels_var = np.reshape(labels_acc[:, :, lev_idx], (-1, 64, 128))
		labels_var = labels_acc[:, lev_idx]
		# labels_var = getattr(labels, var).values
		lev_idx += 1
		error = preds_var - labels_var
		rmse = np.sqrt(((error)**2 * weights_lat).mean())
		rmse_dict[var] = rmse
	return rmse_dict

def create_predictions(model, dataloader, dg, device):
	preds = None
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			# tqdm.write("Testing batch {}".format(i))
			# 		inputs, labels = data
			# 		inputs, labels = torch.tensor(inputs), torch.tensor(labels)
			# 		inputs, labels = inputs.to(device), labels.to(device)
			# 		inputs, labels = inputs.permute(0, 2, 3, 1), labels.permute(0, 2, 3, 1)
			# 		inputs, labels = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1]), labels.reshape(labels.shape[0], -1, labels.shape[-1])
			# 		outputs = model(inputs)				

			inputs, labels = process_data(data, device)
			for forecast_num in range(dg.lead_time // 6):
				inputs = model(inputs)
			outputs = model(inputs)
			outputs = outputs.cpu().detach().numpy()
			if preds is None:
				preds = outputs
			else:
				preds = np.concatenate((preds, outputs), axis=0)

	preds = preds.reshape(-1, 64, 128, 2)
	preds = preds.transpose(0, 3, 1, 2)
	preds = preds*np.array(dg.std.values).reshape((1,2,1,1)) + np.array(dg.mean.values).reshape((1,2,1,1))
	das = []
	lev_idx = 0
	for var, levels in dg.var_dict.items():
		if levels is None:
			das.append(xr.DataArray(
				preds[:, lev_idx, :, :],
				dims=['time', 'lat', 'lon'],
				coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
				name=var
			))
			lev_idx += 1
		else:
			nlevs = len(levels)
			das.append(xr.DataArray(
				preds[:, lev_idx:lev_idx + nlevs, :, :],
				dims=['time', 'lat', 'lon', 'level'],
				coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon, 'level': levels},
				name=var
			))
			lev_idx += nlevs
	return xr.merge(das)

def create_predictions_analysis(model, dataloader, dg, device, valid):
	preds = create_predictions(model, dataloader, dg, device)
	error = preds - valid
	valid.to_netcdf('/home/scratch/rsaxena2/valid_48h.nc')
	error.to_netcdf('/home/scratch/rsaxena2/model_error_48h.nc')
	return

def process_data(data, device):
	inputs, labels = data
	inputs, labels = torch.tensor(inputs), torch.tensor(labels)
	inputs, labels = inputs.to(device), labels.to(device)
	inputs, labels = inputs.permute(0, 2, 3, 1), labels.permute(0, 2, 3, 1)
	inputs, labels = inputs.reshape(inputs.shape[0], -1, inputs.shape[-1]), labels.reshape(labels.shape[0], -1, labels.shape[-1])
	return inputs, labels

@hydra.main(config_path='/zfsauton2/home/rohans2/pgm_project/graph_weather/config', config_name='config')
def main(cfg: DictConfig):
	writer = SummaryWriter()

	vars = ['z', 't']

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	z = xr.open_mfdataset(f'{cfg.data_dir}/geopotential/*.nc', combine='by_coords')
	t = xr.open_mfdataset(f'{cfg.data_dir}/temperature/*.nc', combine='by_coords')
	ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

	ds_train = ds.sel(time=slice(cfg.train_start, cfg.train_end))
	# ds_valid = ds.sel(time=slice(*valid_years))
	ds_test = ds.sel(time=slice(cfg.test_start, cfg.test_end))

	dic = {var: None for var in vars}
	dg_train = DataGenerator(ds_train, dic, cfg.lead_time, batch_size=cfg.batch_size)
	# dg_valid = DataGenerator(ds_valid, dic, lead_time, batch_size=batch_size, mean=dg_train.mean,
							#  std=dg_train.std, shuffle=False)
	dg_test =  DataGenerator(ds_test, dic, cfg.lead_time, batch_size=cfg.test_batch_size, mean=dg_train.mean,
							 std=dg_train.std, shuffle=False)

	training_loader = torch.utils.data.DataLoader(dg_train, batch_size=cfg.batch_size)
	# validation_loader = torch.utils.data.DataLoader(dg_valid, batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(dg_test, batch_size=cfg.test_batch_size)
	
	lat_lons = [(lat, lon) for lat in ds_train.lat.values for lon in ds_train.lon.values]
	# for lat in range(-90, 90, 1):
		# for lon in range(0, 360, 1):
			# lat_lons.append((lat, lon))
	feature_variances = []
	for var in range(605):
		feature_variances.append(0.0)

	criterion = NormalizedMSELoss(
		lat_lons=lat_lons, feature_variance=torch.zeros((78,)), device=device
	).to(device)
	
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

	if cfg.ckpt_path is not None:
		model.load_state_dict(torch.load(cfg.ckpt_path))
		print(f'Loaded model from {cfg.ckpt_path}')
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

	if not cfg.inference:
		z500_valid_train = load_test_data(f'{cfg.data_dir}/geopotential/', 'z', years=slice(cfg.train_start, cfg.train_end))
		t850_valid_train = load_test_data(f'{cfg.data_dir}/temperature/', 't', years=slice(cfg.train_start, cfg.train_end))
		valid_train = xr.merge([z500_valid_train, t850_valid_train], compat='override')
	
	z500_valid_test = load_test_data(f'{cfg.data_dir}/geopotential/', 'z', years=slice(cfg.test_start, cfg.test_end))
	t850_valid_test = load_test_data(f'{cfg.data_dir}/temperature/', 't', years=slice(cfg.test_start, cfg.test_end))
	valid_test = xr.merge([z500_valid_test, t850_valid_test], compat='override')

	if cfg.inference:
		# print(f'Creating iterative predictions/RMSE for lead time {cfg.lead_time} on test set')
		# rmse_dict = create_iterative_forecast_rmse(model, test_loader, dg_test, device, valid_test)
		# print(f'RMSE test dict: ', rmse_dict)

		create_predictions_analysis(model, test_loader, dg_test, device, valid_test)

		exit()

	for epoch in range(cfg.epochs):
		#Validation
		if epoch % 10 == 0:
			# print(f'Creating predictions/RMSE for lead time {lead_time} on train set')
			# train_rmse_dict = create_predictions_rmse(model, training_loader, dg_train, device, valid_train)
			# print(f'RMSE train dict: ', train_rmse_dict)

			print(f'Creating predictions/RMSE for lead time {cfg.lead_time} on test set')
			test_rmse_dict = create_predictions_rmse(model, test_loader, dg_test, device, valid_test)
			print(f'RMSE test dict: ', test_rmse_dict)
			writer.add_scalar('RMSE/Test Z500', test_rmse_dict['z'], epoch)
			writer.add_scalar('RMSE/Test T850', test_rmse_dict['t'], epoch)
		if epoch % 20 == 0:
			print(f'Creating predictions/RMSE for lead time {cfg.lead_time} on train set')
			train_rmse_dict = create_predictions_rmse(model, training_loader, dg_train, device, valid_train)
			print(f'RMSE train dict: ', train_rmse_dict)
			writer.add_scalar('RMSE/Train Z500', train_rmse_dict['z'], epoch)
			writer.add_scalar('RMSE/Train T850', train_rmse_dict['t'], epoch)

			# print('done')
			# results = results.to_dict()
			# print(results)

			# print(f'Creating predictions for lead time {lead_time} on test set')
			# pred = create_predictions(model, test_loader, dg_test, device)
			# print('Computing RMSE in real units...')
			# results = compute_weighted_rmse(pred, valid_test).load()
			# print('done')

		running_loss = 0.0
		print(f"Epoch {epoch}")
		model.train()
		for i, data in enumerate(tqdm(training_loader)):
			start = time.time()
			inputs, labels = process_data(data, device)

			optimizer.zero_grad()

			outputs = model(inputs)

			# if epoch == 0:
				# print(outputs)
				# print(labels)
				# print('-'*50)
				# exit()
			# loss = torch.square(outputs).mean()				
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 20 == 0:
				tqdm.write(f'Running loss: {running_loss/(i+1)}')
		print(f"Epoch {epoch} loss: {running_loss/(i+1)} time: {time.time() - start}")
		writer.add_scalar('Loss/train', running_loss/(i+1), epoch)
		if epoch % 50 == 0:
			print(f'Saving model weights: gnn_{epoch}.pt')
			torch.save(model.state_dict(), f'gnn_{epoch}.pt')
	# save_dir = '/home/scratch/rohans2/gnn_saved'

	# model_save_fn = os.path.join(save_dir, f'gnn_lead{cfg.lead_time}h_epochs{cfg.epochs}.h5')
	model_save_fn = 'gnn.pt'
	print(f'Saving model weights: {model_save_fn}')
	torch.save(model.state_dict(), f'{model_save_fn}')

	#Create Predictions
	# print(f'Creating predictions for lead time {cfg.lead_time}...')
	# pred = create_predictions(model, test_loader, dg_test, device)
	# pred_save_fn = os.path.join(save_dir, f'gnn_lead{cfg.lead_time}h_epochs{cfg.epochs}.nc')
	# print(f'Saving predictions: {pred_save_fn}')
	# pred.to_netcdf(pred_save_fn)

	#Print score in real units
	# print('Computing RMSE in real units...')
	# results_save_fn = os.path.join(save_dir, f'gnn_lead{cfg.lead_time}h_epochs{cfg.epochs}_results.json')
	# z500_valid = load_test_data('/home/scratch/rohans2/processed/geopotential/', 'z')
	# t850_valid = load_test_data('/home/scratch/rohans2/processed/temperature/', 't')
	# valid = xr.merge([z500_valid, t850_valid], compat='override')
	# print(f'Lead time = {cfg.lead_time}')
	# results = compute_weighted_rmse(pred, valid).load()
	# results = results.to_dict()
	# results_dict = {}
	# results_dict['z'] = results['data_vars']['z']['data']
	# results_dict['t'] = results['data_vars']['t']['data']
	# with open(results_save_fn, 'w') as f:
		# f.write(json.dumps(results_dict))

	# out = model(features)
	# criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
	# loss = criterion(out, features)
	# loss.backward()


if __name__ == "__main__":
	main()