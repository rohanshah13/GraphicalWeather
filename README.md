# Probabilistic Graphical Models for Weather Forecasting

## Data
In download.py, update the variables to set the download destination, resolution and atmospheric variable
Then run ```python download.py```

## Diffusion Models
After downloading the data, run ```python train_diffuser.py```. This script automatically trains a diffusion model, generates samples for the test set as well as the evaluation plots and calculates the test metrics.
The syntax to run is as follows: ```python train_diffuser.py [--run_name] [--epochs] [--seed] [--lead_time] [--batch_size] [--num_classes] [--dataset_path] [--device] [--lr] [--noise_steps] [--train_years] [--valid_years] [--test_years]```. Descriptions of each argument can be found within the code or by running ```python train_diffuser.py --help```.


## Graph Neural Networks
Change directory to graph_weather
Update config/config.yaml to set the desired hyperparameters for training
In particular, update the data_dir to the path on your local device
Then run ```python train/train.py```
If inference is False, the model will train, else it will run evaluation on the test data
