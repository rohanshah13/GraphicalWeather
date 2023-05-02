from climate_learn.data import download

root = "/home/scratch/rohans2/pgm_project"
source = "weatherbench"
dataset = "era5"
resolution = "2.8125"
variable = "temperature"

download(root=root, source=source, dataset=dataset, resolution=resolution, variable=variable)