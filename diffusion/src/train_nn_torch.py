from score import *
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from configargparse import ArgParser
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import json


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


class PeriodicPadding2D(nn.Module):
	def __init__(self, pad_width):
		super().__init__()
		self.pad_width = pad_width

	def forward(self, inputs):
		if self.pad_width == 0:
			return inputs
		inputs_padded = torch.cat((inputs[:, :, :, -self.pad_width:], inputs, inputs[:, :, :, :self.pad_width]), dim=3)
		inputs_padded = torch.nn.functional.pad(inputs_padded, (0, 0, self.pad_width, self.pad_width), 'constant', 0)
		return inputs_padded


class PeriodicConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, conv_kwargs={}):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.conv_kwargs = conv_kwargs
		if type(kernel_size) is not int:
			assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
			kernel_size = kernel_size[0]
		pad_width = (kernel_size - 1) // 2
		self.padding = PeriodicPadding2D(pad_width)
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0)

	def forward(self, inputs):
		inputs = self.padding(inputs)
		inputs = self.conv(inputs)
		return inputs

class PeriodicCNN(nn.Module):
	def __init__(self, filters, kernels, input_channels):
		super().__init__()
		layers = [PeriodicConv2D(input_channels, filters[0], kernels[0])]
		layers.append(nn.ELU())
		for i in range(1, len(filters) - 1):
			layers.append(PeriodicConv2D(filters[i-1], filters[i], kernels[i]))
			layers.append(nn.ELU())
		layers.append(PeriodicConv2D(filters[-2], filters[-1], kernels[-1]))
		self.layers = nn.Sequential(*layers)

	def forward(self, inputs):
		print('input shape: ', inputs.shape)
		print('output shape: ', self.layers(inputs).shape)
		return self.layers(inputs)


def create_predictions(model, dataloader, dg, device):
	preds = None
	model.train(False)
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			inputs, labels = data
			inputs, labels = torch.tensor(inputs).to(device), torch.tensor(labels).to(device)
			outputs = model(inputs)
			outputs = outputs.cpu().detach().numpy()
			if preds is None:
				preds = outputs
			else:
				preds = np.concatenate((preds, outputs), axis=0)

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


def main(datadir, vars, filters, kernels, lr, activation, dr, batch_size, patience, save_dir,
		 train_years, valid_years, test_years, lead_time, gpu, iterative, num_epochs):
	np.random.seed(0)
	torch.manual_seed(0)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	writer = SummaryWriter()
	# os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
	# Open dataset and create data generators
	# TODO: Flexible input data
	z = xr.open_mfdataset(f'{datadir}/geopotential_500/*.nc', combine='by_coords')
	t = xr.open_mfdataset(f'{datadir}/temperature_850/*.nc', combine='by_coords')
	ds = xr.merge([z, t], compat='override')  # Override level. discarded later anyway.

	# TODO: Flexible valid split
	ds_train = ds.sel(time=slice(*train_years))
	ds_valid = ds.sel(time=slice(*valid_years))
	ds_test = ds.sel(time=slice(*test_years))

	dic = {var: None for var in vars}
	dg_train = DataGenerator(ds_train, dic, lead_time, batch_size=batch_size)
	dg_valid = DataGenerator(ds_valid, dic, lead_time, batch_size=batch_size, mean=dg_train.mean,
							 std=dg_train.std, shuffle=False)
	dg_test =  DataGenerator(ds_test, dic, lead_time, batch_size=batch_size, mean=dg_train.mean,
							 std=dg_train.std, shuffle=False)
	# print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')
	training_loader = torch.utils.data.DataLoader(dg_train, batch_size=batch_size)
	validation_loader = torch.utils.data.DataLoader(dg_valid, batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(dg_test, batch_size=batch_size)

	model = PeriodicCNN(filters, kernels, 2)
	model.to(device)
	optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
	loss_fn = torch.nn.MSELoss()

	early_stopping = EarlyStopping(patience=patience, verbose=True)
	for epoch in tqdm(range(num_epochs)):
		running_loss = 0
		model.train(True)
		for i, data in tqdm(enumerate(training_loader), total=len(training_loader)):
			# if i % 100 == 0:
			# 	print(i)
			inputs, labels = data
			inputs, labels = torch.tensor(inputs), torch.tensor(labels)
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = loss_fn(outputs, labels)
			loss.backward()

			optimizer.step()

			running_loss += loss
		avg_loss = running_loss / (i + 1)

		#Evaluate on validation set
		running_vloss = 0
		model.train(False)
		with torch.no_grad():
			for i, vdata in enumerate(validation_loader):
				vinputs, vlabels = vdata
				vinputs, vlabels = torch.tensor(vinputs), torch.tensor(vlabels)
				vinputs, vlabels = vinputs.to(device), vlabels.to(device)
				voutputs = model(vinputs)
				vloss = loss_fn(voutputs, vlabels)
				running_vloss += vloss
		avg_vloss = running_vloss / (i+1)

		print(f'Training Loss for epoch {epoch} = {avg_loss}')
		print(f'Validation Loss for epoch {epoch} = {avg_vloss}')
		writer.add_scalars('Training vs. Validation Loss',
						   {'Training': avg_loss, 'Validation': avg_vloss},
						   epoch + 1)
		writer.flush()

		early_stopping(avg_vloss, model)

		if early_stopping.early_stop:
			print("Early stopping")
			break

	model_save_fn = os.path.join(save_dir, f'cnn_3d_lead{lead_time}_epochs{num_epochs}.h5')
	print(f'Saving model weights: {model_save_fn}')
	torch.save(model.state_dict(), f'{model_save_fn}')

	#Create Predictions
	print(f'Creating predictions for lead time {lead_time}...')
	pred = create_predictions(model, test_loader, dg_test, device)
	pred_save_fn = os.path.join(save_dir, f'cnn_3d_lead{lead_time}_epochs{num_epochs}.nc')
	print(f'Saving predictions: {pred_save_fn}')
	pred.to_netcdf(pred_save_fn)

	#Print score in real units
	print('Computing RMSE in real units...')
	results_save_fn = os.path.join(save_dir, f'cnn_3d_lead{lead_time}_epochs{num_epochs}_results.json')
	z500_valid = load_test_data(f'{datadir}geopotential_500', 'z')
	t850_valid = load_test_data(f'{datadir}temperature_850', 't')
	valid = xr.merge([z500_valid, t850_valid], compat='override')
	print(f'Lead time = {lead_time}')
	results = compute_weighted_rmse(pred, valid).load()
	results = results.to_dict()
	results_dict = {}
	results_dict['z'] = results['data_vars']['z']['data']
	results_dict['t'] = results['data_vars']['t']['data']
	with open(results_save_fn, 'w') as f:
		f.write(json.dumps(results_dict))

if __name__ == '__main__':
	p = ArgParser()
	p.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
	p.add_argument('--datadir', type=str, required=True, help='Path to data')
	p.add_argument('--save_dir', type=str, required=True, help='Path to save model and predictions')
	p.add_argument('--vars', type=str, nargs='+', required=True, help='Variables')
	p.add_argument('--filters', type=int, nargs='+', required=True, help='Filters for each layer')
	p.add_argument('--kernels', type=int, nargs='+', required=True, help='Kernel size for each layer')
	p.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
	p.add_argument('--iterative', type=bool, default=False, help='Is iterative forecast')
	p.add_argument('--iterative_max_lead_time', type=int, default=5*24, help='Max lead time for iterative forecasts')
	p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
	p.add_argument('--activation', type=str, default='elu', help='Activation function')
	p.add_argument('--dr', type=float, default=0, help='Dropout rate')
	p.add_argument('--batch_size', type=int, default=128, help='batch_size')
	p.add_argument('--patience', type=int, default=3, help='Early stopping patience')
	p.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
	p.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'), help='Start/stop years for validation')
	p.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
	p.add_argument('--gpu', type=int, default=0, help='Which GPU')
	p.add_argument('--num_epochs', type=int, default=100)
	args = p.parse_args()

	main(
		datadir=args.datadir,
		vars=args.vars,
		filters=args.filters,
		kernels=args.kernels,
		lr=args.lr,
		activation=args.activation,
		dr=args.dr,
		batch_size=args.batch_size,
		patience=args.patience,
		save_dir=args.save_dir,
		train_years=args.train_years,
		valid_years=args.valid_years,
		test_years=args.test_years,
		lead_time=args.lead_time,
		gpu=args.gpu,
		iterative=args.iterative,
		num_epochs=args.num_epochs,
	)