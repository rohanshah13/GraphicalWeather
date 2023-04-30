"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import xarray as xr
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from data_utils import *
from modules import UNet_conditional, EMA

config = SimpleNamespace(
    run_name="DDPM_conditional",
    epochs=50,
    noise_steps=1000,
    seed=42,
    batch_size=64,
    num_classes=10,
    dataset_path = '/home/scratch/vdas/weatherbench/data',
    train_folder="train",
    val_folder="test",
    device="cuda",
    slice_size=1,
    do_validation=False,
    fp16=True,
    log_every_epoch=5,
    num_workers=10,
    lr=5e-3)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def compute_weighted_rmse(da_fc, da_true):
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(np.linspace(-90, 90, 32))).reshape(-1, 1)
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean())
    return rmse

def plot_images_weatherbench_generate(gt, images, idx):
    plt.rcParams["figure.figsize"] = (60, 20)
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(gt[0, :, :], cmap='coolwarm')
    ax[0, 1].imshow(images[0, :, :], cmap='coolwarm')
    ax[0, 2].imshow(np.abs(images[0, :, :] - gt[0, :, :]), cmap='coolwarm')
    ax[1, 0].imshow(gt[1, :, :], cmap='coolwarm')
    ax[1, 1].imshow(images[1, :, :], cmap='coolwarm')
    ax[1, 2].imshow(np.abs(images[1, :, :] - gt[1, :, :]), cmap='coolwarm')
    ax[0, 0].set_title('Ground Truth', fontsize=40)
    ax[0, 1].set_title('Prediction', fontsize=40)
    ax[0, 2].set_title('Error', fontsize=40)
    ax[0, 0].set_ylabel(r'Z500 [$m^{2}s^{-2}$]', fontsize=40)
    ax[1, 0].set_ylabel('T850 [K]', fontsize=40)
    # Add colorbars to all figures
    fig.colorbar(ax[0, 0].imshow(gt[0, :, :], cmap='coolwarm'), ax=ax[0, 0], shrink=0.79)
    fig.colorbar(ax[0, 1].imshow(images[0, :, :], cmap='coolwarm'), ax=ax[0, 1], shrink=0.79)
    fig.colorbar(ax[0, 2].imshow(np.abs(images[0, :, :] - gt[0, :, :]), cmap='coolwarm'), ax=ax[0, 2], shrink=0.79)
    fig.colorbar(ax[1, 0].imshow(gt[1, :, :], cmap='coolwarm'), ax=ax[1, 0], shrink=0.79)
    fig.colorbar(ax[1, 1].imshow(images[1, :, :], cmap='coolwarm'), ax=ax[1, 1], shrink=0.79)
    fig.colorbar(ax[1, 2].imshow(np.abs(images[1, :, :] - gt[1, :, :]), cmap='coolwarm'), ax=ax[1, 2], shrink=0.79)
    # Remove ticks from all figures
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    if not os.path.exists(f'/home/scratch/vdas/weatherbench/results/images/'):
        os.makedirs(f'/home/scratch/vdas/weatherbench/results/images/')
    plt.savefig(f'/home/scratch/vdas/weatherbench/results/images/{idx}.png')
    plt.clf()
    plt.close()


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, num_classes=10, c_in=2, c_out=2,
                 device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, 32, 64)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                # print('sample: ', x.shape, t.shape, labels.shape)
                predicted_noise = model(x, t, labels)
                # if cfg_scale > 0:
                #     uncond_predicted_noise = model(x, t, None)
                #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        # x = x.clamp(-1, 1)
        # x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (labels, images) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                # if np.random.random() < 0.1:
                #     labels = None
                # print('fit: ', x_t.shape, t.shape, labels.shape)
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({"train_mse": loss.item(),
                           "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()

    def log_images(self, ep, dataloader, num=None, save=False):
        i = 0
        for image, context in tqdm(dataloader):
            if num is not None and i >= num:
                break
            i += 1
            image = image.to(self.device)
            context = context.to(self.device)
            sampled_images = self.sample(use_ema=False, labels=context)
            for idx, (img, si) in enumerate(zip(image, sampled_images)):
                if save:
                    if not os.path.exists(f"/home/scratch/vdas/weatherbench/outputs/{ep}"):
                        os.mkdir(f"/home/scratch/vdas/weatherbench/outputs/{ep}")
                    np.save(f"/home/scratch/vdas/weatherbench/outputs/{ep}/{idx}.npy", si.cpu().numpy())
                plot_images_weatherbench(img, si, ep, idx)

    def evaluate(self, dataloader):
        z500mses = []
        t850mses = []
        i = 0
        # sample_paths = os.listdir("/home/scratch/vdas/weatherbench/outputs/6h/samples")
        sample_paths = os.listdir("/home/scratch/vdas/weatherbench/outputs/more_samples")
        ks = [0.001, 0.01, 0.1, 0.2, 0.3]
        # ks = [0.001, 0.01]
        containmentsz500 = [[] for _ in ks]
        containmentst850 = [[] for _ in ks]
        for context, image in tqdm(dataloader, total=len(dataloader)):
            image = image.to(self.device)
            context = context.to(self.device)
            for idx, img in tqdm(enumerate(image)):
                # sampled_images = np.array([np.load(f"/home/scratch/vdas/weatherbench/outputs/6h/samples/{s}") for s in
                sampled_images = np.array([np.load(f"/home/scratch/vdas/weatherbench/outputs/more_samples/{s}") for s in
                                  sample_paths if s.endswith(f"{i}.npy")])
                if len(sampled_images) == 0:
                    continue
                img = img.cpu().numpy() * np.array(dataloader.dataset.std).reshape((2,1,1)) + np.array(dataloader.dataset.mean).reshape((2,1,1))
                print(np.max(img), np.min(img))
                # sampled_images = (sampled_images - np.array(dataloader.dataset.mean).reshape((1,2,1,1))) / np.array(dataloader.dataset.std).reshape((1,2,1,1))

                sampled_images = np.mean(sampled_images, axis=0)
                sampled_images_std = np.std(sampled_images, axis=0)
                for k_idx, k in enumerate(ks):
                    lower = img - k * sampled_images_std
                    upper = img + k * sampled_images_std
                    containmentsz500[k_idx].append(np.mean((sampled_images[0] >= lower[0]) & (sampled_images[0] <= upper[0])))
                    containmentst850[k_idx].append(np.mean((sampled_images[1] >= lower[1]) & (sampled_images[1] <= upper[1])))
                z500mses.append(compute_weighted_rmse(img[0], sampled_images[0]))
                t850mses.append(compute_weighted_rmse(img[1], sampled_images[1]))
                i += 1
                plot_images_weatherbench_generate(img, sampled_images, i)
        containmentsz500 = [np.mean(containment) for containment in containmentsz500]
        containmentst850 = [np.mean(containment) for containment in containmentst850]
        # Plot both containments with a dot at each point, and add a legend
        plt.plot(ks, containmentsz500, label='Z500', marker='o')
        plt.plot(ks, containmentst850, label='T850', marker='o')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('Fractions of pixels within k std')
        plt.savefig('containment.png')
        plt.close()
        print(f"Z500 RMSE: {np.mean(z500mses)}")
        print(f"T850 RMSE: {np.mean(t850mses)}")

    def load(self, model_ckpt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_ckpt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_ckpt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional",
                            metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr,
                                                       steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)

            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})

            # log predicitons
            if epoch % args.log_every_epoch == 0:
            #     print("Saving model")
            #     self.log_images(epoch, self.val_dataloader, num = 10)

        # save model
                self.save_model(run_name=args.run_name, epoch=epoch)
        self.log_images(epoch, self.test_dataloader, save=True)


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--vars', type=str, nargs='+', default=('z', 't'), help='Variables')
    parser.add_argument('--lead_time', type=int, default=6, help='Forecast lead time')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
    parser.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'),
                   help='Start/stop years for validation')
    parser.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, num_classes=config.num_classes, c_in=2, c_out=2)
    with wandb.init(project="test", group="train", config=config):
        diffuser.prepare(config)
        diffuser.load(f'models/{config.run_name}')
        diffuser.fit(config)