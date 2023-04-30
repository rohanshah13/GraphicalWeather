# Probabilistic Graphical Models for Weather Forecasting

## Diffusion Models
After downloading the data, run ```python train_diffuser.py```. This script automatically trains a diffusion model, generates samples for the test set as well as the evaluation plots and calculates the test metrics.
The syntax to run is as follows: ```python train_diffuser.py [--run_name] [--epochs] [--seed] [--lead_time] [--batch_size] [--num_classes] [--dataset_path] [--device] [--lr] [--noise_steps] [--train_years] [--valid_years] [--test_years]```. Descriptions of each argument can be found within the code or by running ```python train_diffuser.py --help```.
