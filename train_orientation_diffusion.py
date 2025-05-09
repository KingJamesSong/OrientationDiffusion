import os
import copy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fid_evaluation import FIDEvaluation
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, EMA
import logging
import numpy
from torch.utils.tensorboard import SummaryWriter
from patch_ae import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.autograd import grad

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def wrapped_gaussian_score(y, mu, sigma, K=3):
    """
    Compute the score function for a wrapped Gaussian on [-pi, pi].

    Args:
      y (torch.Tensor): Angles in [-pi, pi], shape [*], can be batch.
      mu (torch.Tensor): Mean parameter (same shape as y or broadcastable).
      sigma (float or torch.Tensor): Std parameter (scalar or broadcastable).
      K (int): Truncation level for the infinite sum.

    Returns:
      torch.Tensor: Score function values at each y, same shape as y.
    """
    k_range = torch.arange(-K, K+1, device=y.device).float()
    y_2D = y.unsqueeze(-1)                # [batch,1]
    mu_2D = mu.unsqueeze(-1)              # [batch,1]
    sigma_2D = sigma.unsqueeze(-1)  # [batch,1]
    k_2D = k_range.unsqueeze(0)           # [1, 2K+1]

    # Shift: (mu - 2πk)
    shift = mu_2D - 2*torch.pi*k_2D        # [batch, 2K+1]
    # difference: y - shift
    diff = y_2D - shift                   # [batch, 2K+1]

    # exponent
    exp_terms = torch.exp(-0.5 * (diff**2) / (sigma_2D**2))  # [batch, 2K+1]

    # denominator = sum of exponent
    denom = torch.sum(exp_terms, dim=-1)  # shape [batch]

    # numerator = sum( diff * exp(...) )
    numer = torch.sum(diff * exp_terms, dim=-1)  # shape [batch]

    # Finally the score
    # note the -1/sigma^2 factor outside
    # score = - (1 / sigma^2) * numer / denom
    # ensure sigma is broadcast
    if isinstance(sigma, float):
        sigma_sq = sigma**2
    else:
        sigma_sq = sigma**2
        # if sigma is a tensor, ensure shape matches y appropriately

    score_val = - (1.0 / sigma_sq) * (numer / denom)

    return score_val

def map_to_phase(x):
    x = x * 0.9*numpy.pi
    return x

def phase_modulate(x):
    x = (x + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return x

def map_to_image(x):
    x = x / (0.9*numpy.pi)
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

def angle_space(score, angle):
    C = angle.size(1)
    sin_score = score[:,:C,:,:]
    cos_score = score[:,C:,:,:]
    score = -sin_score * torch.sin(angle) + cos_score * torch.cos(angle)
    return score

class OrientationDiffusion:
    def __init__(self, noise_steps=100, noise_start=1e-4, noise_end=0.1, coupling_start=3e-5, coupling_end=0.03, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        self.coupling_st = torch.linspace(coupling_start, coupling_end, self.noise_steps).to(self.device)
        #self.gauss = torch.distributions.normal.Normal(0.0,1.0)
        self.noise_st = self.prepare_noise_schedule(noise_start, noise_end)
        self.ref_phase = 0.

    def prepare_noise_schedule(self, noise_start, noise_end):
        return torch.linspace(noise_start, noise_end, self.noise_steps).sqrt().to(self.device)

    def coupling_noise(self, x):
        w = torch.randn_like(x)
        #deltax = x.unsqueeze(4).unsqueeze(4).repeat(1, 1, 1, 1, H, W)
        complex_phases = torch.complex(x.cos(),x.sin())
        #complex_phases = torch.exp(1j * x)
        mean_phase_vector = torch.mean(complex_phases,dim=[1,2,3],keepdim=True)
        order_phase = torch.atan2(mean_phase_vector.imag, mean_phase_vector.real)
        order_parameter = torch.sqrt(mean_phase_vector.imag**2 + mean_phase_vector.real**2)
        #order_parameter = mean_phase_vector.abs()
        deltax = order_parameter * torch.sin(order_phase - x) + 1.5*torch.sin(self.ref_phase - x)
        #deltax = torch.mean(torch.sin(deltax - x.unsqueeze(4).unsqueeze(4)), dim=[4, 5])
        return w, deltax, order_parameter, order_phase

    def coupling_noise_local(self, x, M=5):
        w = torch.randn_like(x)
        complex_phases = torch.exp(1j * x)
        kernel = torch.ones(1, 1, 2 * M + 1, 2 * M + 1) / ((2 * M + 1) ** 2)
        kernel = kernel.repeat(3,1,1,1).to(x)
        # Apply circular padding to handle periodic boundary conditions
        complex_padded = F.pad(complex_phases, (M, M, M, M), mode='circular')
        order_phase = torch.atan2(complex_padded.imag, complex_padded.real)
        order_parameter = torch.sqrt(complex_padded.imag ** 2 + complex_padded.real ** 2)
        order_parameter = F.conv2d(order_parameter, kernel, groups=3)
        order_phase = F.conv2d(order_phase, kernel, groups=3)
        deltax = order_parameter * torch.sin(order_phase - x) +  torch.sin(self.ref_phase - x)
        return w, deltax, order_parameter, order_phase

    def noise_images(self, x, t):
        for i in range(0, t.max()):
            mask = (t > i).nonzero().squeeze(dim=1)
            w, deltax, order_parameter, order_phase = self.coupling_noise(x[mask])
            x[mask] = x[mask] + self.coupling_st[i] * deltax + self.noise_st[i] * w
            x[mask] = phase_modulate(x[mask])
        return x

    def sample_noise(self, model, step, size, loc, concentration):
        dist = torch.distributions.von_mises.VonMises(torch.Tensor([loc]),torch.Tensor([concentration]))
        x = dist.sample(size).squeeze().to(self.device)
        with torch.no_grad():
            for i in tqdm(reversed(range(step, self.noise_steps)), position=0):
                t = (torch.ones(size[0]) * i).long().to(self.device)
                w, deltax, order_parameter, order_phase = self.coupling_noise(x)
                score = model(x,t)
                x = x - self.coupling_st*deltax + (self.noise_st[t]**2)*score + self.noise_st[t]*w
                #x = phase_modulate(x)
        return x, score

    def sample_timesteps(self,n, epoch):
        return torch.randint(low=0, high=self.noise_steps-1, size=(n,))

    def sample(self, model, n, run_name=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            dist = torch.distributions.von_mises.VonMises(torch.Tensor([0.0]), torch.Tensor([((0.50+1.5)*self.coupling_st[-1] ) / (0.5*self.noise_st[-1]**2)]))
            x = dist.sample((n, 3, self.img_size, self.img_size)).squeeze().to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                w, deltax,order_parameter, order_phase = self.coupling_noise(x)
                score = model(x, t)
                score = angle_space(score, x)
                x = x - self.coupling_st[i]*deltax + (self.noise_st[i]**2) * score + self.noise_st[i]*w
                x = phase_modulate(x)
                if run_name != None:
                    save_images(map_to_image(x),os.path.join("recon_images", run_name, f"{self.noise_steps - i}.jpg"))
        model.train()
        return x

    def sample_image(self,model, x):
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, self.noise_steps-1), position=0):
                w, deltax, order_parameter, order_phase = self.coupling_noise(x)
                x = x + self.coupling_st[i] * deltax + self.noise_st[i] * w
                x = phase_modulate(x)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(1) * i).long().to(self.device)
                score = model(x, t)
                score = angle_space(score, x)
                w, deltax, order_parameter, order_phase = self.coupling_noise(x)
                x = x - self.coupling_st[i]*deltax + (self.noise_st[i]**2) * score + self.noise_st[i]*w
                x = phase_modulate(x)
        model.train()
        return x

def train_dsm(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, test_dataloader = get_data(args)
    model = UNet(img_size=args.image_size).to(device)
    #model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs // 10, eta_min=1e-5)
    mse = nn.MSELoss()
    diffusion = OrientationDiffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            #map to phase space
            images = map_to_phase(images)
            t = diffusion.sample_timesteps(images.size(0), epoch).to(device)   
            loss = 0.0
            exp_sam = 1
            exp_sam2 = 5
            for sample in range(exp_sam):
                if sample == 0:
                    x_t = diffusion.noise_images(images, t)
                    x_t = x_t.unsqueeze(0)
                else:
                    temp_x = diffusion.noise_images(images, t)
                    x_t = torch.cat([x_t, temp_x.unsqueeze(0)],dim=0)
            for sample in range(exp_sam):
                for sample2 in range(exp_sam2):
                    w, deltax, order_parameter, order_phase = diffusion.coupling_noise(x_t[sample])
                    Fxt = x_t[sample] + diffusion.coupling_st[t][:, None, None, None]  * deltax
                    Fxt = phase_modulate(Fxt)
                    x_t1 = Fxt + diffusion.noise_st[t][:, None, None, None] * w
                    x_t1 = phase_modulate(x_t1)
                    _, deltax_xt1, _, _ = diffusion.coupling_noise(x_t1)
                    predicted_score = model(x_t1, t + 1)
                    predicted_score = angle_space(predicted_score, x_t1)
                    score = wrapped_gaussian_score(x_t1,Fxt,diffusion.noise_st[t][:, None, None, None])
                    loss += mse(predicted_score * diffusion.noise_st[t + 1][:, None, None, None] ** 2 / 2,
                                score * diffusion.noise_st[t + 1][:, None, None, None] ** 2 / 2)
            loss = loss /exp_sam /exp_sam2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        #scheduler.step()
        if epoch%5==0:
            sampled_images = diffusion.sample_image(ema_model, images)
            save_images(map_to_image(sampled_images), os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #save_images(save_phase(sampled_images), os.path.join("results", args.run_name, f"phase_{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))


def validate(args):
    device = args.device
    setup_logging(args.run_name)
    args.batch_size = 80
    train_dataloader, test_dataloader = get_data(args)
    model = UNet(img_size=args.image_size).to(device)
    #load pre-trained model
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
    diffusion = OrientationDiffusion(img_size=args.image_size, device=device)
    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        for sample, (images, _) in enumerate(pbar):
            if sample>0:
                break
            images = images.to(device)
            #map to phase space
            images =map_to_phase(images)
            save_images(map_to_image(images), os.path.join("noise", args.run_name, "-1.jpg"))
            for t in tqdm(range(0, diffusion.noise_steps-1), position=0):
                w, deltax, order_parameter, order_phase = diffusion.coupling_noise(images)
                images = images + diffusion.noise_st[t] * w + diffusion.coupling_st[t] * deltax
                images = phase_modulate(images)
                save_images(map_to_image(images), os.path.join("noise", args.run_name, f"{t}.jpg"))
                #save_images(save_phase(images), os.path.join("phase", args.run_name, f"{t}.jpg"))

            text_save(os.path.join("noise", args.run_name, "./order_parameter.txt"), numpy.array(order_parameter.cpu().view(-1).numpy()))
            text_save(os.path.join("noise", args.run_name, "./order_phase.txt"), numpy.array(order_phase.cpu().view(-1).numpy()))
            
            for i in tqdm(reversed(range(1, diffusion.noise_steps)), position=0):
                t = (torch.ones(1) * i).long().to(images.device)
                predicted_score = model(images, t)
                predicted_score = angle_space(predicted_score, images)
                w, deltax, order_parameter, order_phase = diffusion.coupling_noise(images)
                images = images - diffusion.coupling_st[t]*deltax + (diffusion.noise_st[i]**2) * predicted_score + diffusion.noise_st[i]*w
                images = phase_modulate(images)
                save_images(map_to_image(images), os.path.join("recon_images", args.run_name, f"{diffusion.noise_steps-i}.jpg"))

def validate_noise(args):
    setup_logging(args.run_name)
    device = args.device
    args.batch_size = 80
    train_dataloader, test_dataloader = get_data(args)
    model = UNet(img_size=args.image_size).to(device)
    # load pre-trained model
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
    diffusion = OrientationDiffusion(img_size=args.image_size, device=device)
    pbar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        diffusion.sample(model,args.batch_size,args.run_name)

def validate_fid(args):
    setup_logging(args.run_name)
    device = args.device
    args.batch_size = 512
    train_dataloader, test_dataloader = get_data(args)
    model = UNet(img_size=args.image_size).to(device)
    #load pre-trained model
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
    diffusion = OrientationDiffusion(img_size=args.image_size, device=device)
    fid_eval = FIDEvaluation(batch_size=args.batch_size, dl=test_dataloader, sampler=diffusion, channels=3, model=model,dataset=args.dataset_name)
    fid_eval.load_or_precalc_dataset_stats()
    fid_eval.fid_score_image()
    fid_eval.fid_score_noise()

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "OrientationCIFAR10"
    args.dataset_name = "cifar"
    args.epochs = 1000
    args.batch_size = 128
    args.image_size = 32
    args.patch_size = 8
    args.dataset_path = "/home/"
    args.device = "cuda"
    args.lr = 1e-4
    train_dsm(args)


if __name__ == '__main__':
    launch()
   
