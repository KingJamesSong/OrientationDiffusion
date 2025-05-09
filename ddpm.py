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

def gaussian_score(y, mu, sigma):
    score = (mu - y) / (sigma** 2)
    return score

def generate_edges(image):
    B,C,H,W = image.size()
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(image)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).to(image)
    # Apply convolution to extract gradients
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).float()
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).float()
    grad_x = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    orientation_map = torch.atan2(grad_y, grad_x)
    # Compute edge magnitude
    edge_image = torch.sqrt(grad_x**2 + grad_y**2)
    #edge_max, _ = torch.max(edge_image.view(B, C, H * W), dim=-1, keepdim=True)
    #edge_min, _ = torch.min(edge_image.view(B, C, H * W), dim=-1, keepdim=True)
    #edge_image = (edge_image - edge_image.mean(dim=[2,3],keepdim=True))
    edge_max, _ = torch.max(edge_image.view(B, C, H * W), dim=-1, keepdim=True)
    edge_image = edge_image / edge_max.unsqueeze(-1)
    return edge_image, orientation_map

def generate_laplacian(image):
    B, C, H, W = image.size()
    # Define Sobel filters
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).to(image)
    # Apply convolution to extract gradients
    laplacian = laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1).float()
    images_lap = torch.nn.functional.conv2d(image, laplacian, padding=1)
    return images_lap

def generate_blur(image):
    # Define Sobel filters
    blur = torch.tensor([[1, 1, 1], [1, 8, 1], [1, 8, 1]]).to(image)
    B, C, H, W = image.size()
    # Apply convolution to extract gradients
    blur = blur.unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).float()
    image = torch.nn.functional.conv2d(image, blur, padding=1)
    image = (image - image.mean(dim=[2, 3], keepdim=True))
    img_max, _ = torch.max(image.abs().view(B, C, H * W), dim=-1, keepdim=True)
    image = image / img_max.unsqueeze(-1)
    phase = 2 * torch.arccos(image)
    return image, phase

def generate_orientation(image):
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(image)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).to(image)
    # Apply convolution to extract gradients
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).float()
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(3,3,1,1).float()
    grad_x = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    orientation_map = torch.atan2(grad_y, grad_x)
    return orientation_map

def map_to_phase(x):
    #x, _, _ = image_ae.encode_phase(x)
    x = x * 0.9*numpy.pi
    return x

def phase_modulate(x):
   # while x.max() > numpy.pi or x.min() < -numpy.pi:
   #     x = torch.where(x >= numpy.pi, x - 2*numpy.pi, x)
   #     x = torch.where(x < -numpy.pi, x + 2*numpy.pi, x)
    #x = torch.where(x > 1., x - 2., x)
    #x = torch.where(x < -1., x + 2., x)
    x = (x + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return x

def map_to_image(x):
    #if x.max() > numpy.pi or x.min() < -numpy.pi:
    #    x = torch.where(x > numpy.pi, x - 2 * numpy.pi, x)
    #    x = torch.where(x < -numpy.pi, x + 2 * numpy.pi, x)
    #x = image_ae.decode_phase(x)
    x = x / (0.9*numpy.pi)
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

def kl_div_wg(mu_q, mu_p, sigma_q, K=3, num_samples=10):
    # Expand mu_q, mu_p, sigma_q for multiple samples
    mu_q_expanded = mu_q.unsqueeze(0).repeat(num_samples,1,1,1,1)
    mu_p_expanded = mu_p.unsqueeze(0).repeat(num_samples,1,1,1,1)
    sigma_q_expanded = sigma_q.unsqueeze(0).repeat(num_samples,1,1,1,1)

    # Sample multiple z from q
    eps = torch.randn_like(mu_q_expanded)
    z = mu_q_expanded + eps * sigma_q_expanded

    # Precompute k values: [2K + 1]
    k = torch.arange(-K, K + 1, device=mu_q.device).reshape(1, 1, 1, 1, -1)

    # Compute log q(z)
    delta_q = z.unsqueeze(-1) - mu_q_expanded.unsqueeze(-1) - 2 * numpy.pi * k
    log_terms_q = (
        -0.5 * (delta_q ** 2) / (sigma_q_expanded.unsqueeze(-1) ** 2)
        - torch.log(sigma_q_expanded.unsqueeze(-1) * numpy.sqrt(2 * numpy.pi))
    )
    log_q = torch.logsumexp(log_terms_q, dim=-1)

    # Compute log p(z)
    delta_p = z.unsqueeze(-1) - mu_p_expanded.unsqueeze(-1) - 2 * numpy.pi * k
    log_terms_p = (
        -0.5 * (delta_p ** 2) / (sigma_q_expanded.unsqueeze(-1) ** 2)
        - torch.log(sigma_q_expanded.unsqueeze(-1) * numpy.sqrt(2 * numpy.pi))
    )
    log_p = torch.logsumexp(log_terms_p, dim=-1)

    # KL divergence: E_q[log_q - log_p]
    kl = (log_q - log_p).mean()

    return kl

def angle_space(score, angle):
    C = angle.size(1)
    sin_score = score[:,:C,:,:]
    cos_score = score[:,C:,:,:]
    score = -sin_score * torch.sin(angle) + cos_score * torch.cos(angle)
    return score

def map_to_phase_ae(x,patch_ae):
    patches = patch_ae.extract_patches_non_overlapping(x)
    phase,_,_ = patch_ae.encode_phase(patches)
    phase = patch_ae.reconstruct_from_patches(phase)
    return phase

def map_to_image_ae(phase,patch_ae):
    if phase.max() >  numpy.pi or phase.min() < -numpy.pi:
        phase = torch.where(phase > numpy.pi, phase - 2 * numpy.pi, phase)
        phase = torch.where(phase < -numpy.pi, phase + 2 * numpy.pi, phase)
    phase_patches = patch_ae.extract_patches_non_overlapping(phase)
    patches = patch_ae.decode_phase(phase_patches)
    x = patch_ae.reconstruct_from_patches(patches)
    x = (x+1) / 2
    x = (x * 255).type(torch.uint8)
    return x

class OrientationDiffusion:
    def __init__(self, noise_steps=1000, noise_start=1e-4, noise_end=0.015, coupling_start=3e-5, coupling_end=0.0045, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        self.coupling_st = torch.linspace(coupling_start, coupling_end, self.noise_steps).to(self.device)
        #self.gauss = torch.distributions.normal.Normal(0.0,1.0)
        self.noise_st = self.prepare_noise_schedule(noise_start, noise_end)
        self.ref_phase = 0.#numpy.pi / 2.

    def prepare_noise_schedule(self, noise_start, noise_end):
        return torch.linspace(noise_start, noise_end, self.noise_steps).sqrt().to(self.device)

    def coupling_noise(self, x):
        #B, C, H, W = x.size()
        #w = self.gauss.sample(x.size()).to(x)
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
        #B, C, H, W = x.size()
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
        #if high_end:
        #    return torch.randint(low=1, high=self.noise_steps, size=(1,))
        #else:
        #max_t = int((epoch / 100.) * self.noise_steps)
        #max_t = max(1, max_t)  # Avoid 0
        #max_t = min(self.noise_steps-1, max_t)
        return torch.randint(low=0, high=self.noise_steps-1, size=(n,))

    def sample(self, model, n, run_name=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            #x = self.gauss.sample((n, 3, self.img_size, self.img_size)).to(self.device)
            #x = map_to_phase(x)
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

def train_sb_half(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, test_dataloader = get_data(args)
    model = UNet(img_size=args.image_size).to(device)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
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
            #blur_img, blur_phase = generate_blur(images)
            #map to phase space
            images = map_to_phase(images)
            t = diffusion.sample_timesteps(images.size(0), epoch).to(device)
            #x_t = diffusion.noise_images(images, t)
            #half-bridges
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
                    #mu_q = x_t1-diffusion.coupling_st[t+1][:,None,None,None]*deltax_xt1+diffusion.noise_st[t+1][:,None,None,None]**2*predicted_score
                    #mu_p = x_t1-diffusion.coupling_st[t+1][:,None,None,None]*deltax_xt1+diffusion.noise_st[t+1][:,None,None,None]**2*score
                    #loss += kl_div_wg(mu_q=mu_q, mu_p=mu_p, sigma_q=diffusion.noise_st[t+1][:,None,None,None])
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
            #save_images(save_phase(images), os.path.join("phase", args.run_name, "-1.jpg"))
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
                #save_images(save_phase(images),os.path.join("recon_phases", args.run_name, f"{diffusion.noise_steps - i}.jpg"))
        #text_save(os.path.join("phase", args.run_name, "./all_order.txt"),numpy.array(order.view(-1).numpy()))
        #text_save(os.path.join("phase", args.run_name, "./all_phase.txt"),numpy.array(phase.view(-1).numpy()))
def validate_hierachy(args):
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
            images =map_to_phase(images)
            save_images(map_to_image(images), os.path.join("noise", args.run_name, "-1.jpg"))
            for t in tqdm(range(0, diffusion.noise_steps-1), position=0):
                w, deltax, order_parameter, order_phase = diffusion.coupling_noise(images)
                images = images + diffusion.noise_st[t] * w + diffusion.coupling_st[t] * deltax
                images = phase_modulate(images)
            save_images(map_to_image(images), os.path.join("noise", args.run_name, f"{t}.jpg"))
            for path0 in range(3):
                temp0 = images.clone()
                for i in tqdm(reversed(range(diffusion.noise_steps//4*3, diffusion.noise_steps)), position=0):
                    t = (torch.ones(1) * i).long().to(temp0.device)
                    predicted_score = model(temp0, t)
                    predicted_score = angle_space(predicted_score, temp0)
                    w, deltax, order_parameter, order_phase = diffusion.coupling_noise(temp0)
                    temp0 = temp0 - diffusion.coupling_st[t]*deltax + (diffusion.noise_st[i]**2) * predicted_score + diffusion.noise_st[i]*w
                    temp0 = phase_modulate(temp0)
                save_images(map_to_image(temp0), os.path.join("noise", args.run_name, f"hier0-{path0}.jpg"))
                for path1 in range(3):
                    temp = temp0.clone()
                    for i in tqdm(reversed(range(diffusion.noise_steps//4*2, diffusion.noise_steps//4*3)), position=0):
                        t = (torch.ones(1) * i).long().to(temp.device)
                        predicted_score = model(temp, t)
                        predicted_score = angle_space(predicted_score, temp)
                        w, deltax, order_parameter, order_phase = diffusion.coupling_noise(temp)
                        temp = temp - diffusion.coupling_st[t] * deltax + (
                                    diffusion.noise_st[i] ** 2) * predicted_score + diffusion.noise_st[i] * w
                        temp = phase_modulate(temp)
                    save_images(map_to_image(temp), os.path.join("noise", args.run_name, f"hier1-{path0}{path1}.jpg"))
                    for path2 in range(3):
                        temp2 = temp.clone()
                        for i in tqdm(reversed(range(diffusion.noise_steps//4, diffusion.noise_steps//4*2)),
                                      position=0):
                            t = (torch.ones(1) * i).long().to(temp2.device)
                            predicted_score = model(temp2, t)
                            predicted_score = angle_space(predicted_score, temp2)
                            w, deltax, order_parameter, order_phase = diffusion.coupling_noise(temp2)
                            temp2 = temp2 - diffusion.coupling_st[t] * deltax + (
                                    diffusion.noise_st[i] ** 2) * predicted_score + diffusion.noise_st[i] * w
                            temp2 = phase_modulate(temp2)
                        save_images(map_to_image(temp2), os.path.join("noise", args.run_name, f"hier2-{path0}{path1}{path2}.jpg"))
                        for path3 in range(3):
                            temp3 = temp2.clone()
                            for i in tqdm(reversed(range(1, diffusion.noise_steps // 4)),
                                          position=0):
                                t = (torch.ones(1) * i).long().to(temp3.device)
                                predicted_score = model(temp3, t)
                                predicted_score = angle_space(predicted_score, temp3)
                                w, deltax, order_parameter, order_phase = diffusion.coupling_noise(temp3)
                                temp3 = temp3 - diffusion.coupling_st[t] * deltax + (
                                        diffusion.noise_st[i] ** 2) * predicted_score + diffusion.noise_st[i] * w
                                temp3 = phase_modulate(temp3)
                            save_images(map_to_image(temp3), os.path.join("noise", args.run_name, f"hier3-{path0}{path1}{path2}{path3}.jpg"))

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
    #fid_eval.fid_score_image()
    fid_eval.fid_score_noise()

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "OrientationCIFAR106"
    args.dataset_name = "cifar"
    args.epochs = 1000
    args.batch_size = 128
    args.image_size = 32
    args.patch_size = 8
    args.dataset_path = "/home/yuesong/"
    args.device = "cuda"
    args.lr = 1e-4
    train_sb_half(args)

#10  1e-4-0.1 2e-4-0.2, 0.25, r0.87, imageFID 60.92 noiseFID 60.25 curriculum
#102 1e-4-0.1 1e-4-0.1, 0.5, r0.76 imageFID 57.01 noiseFID 56.77 curriculum
#103  1e-4-0.1 2e-4-0.2, 0.25 BS256, r0.87, imageFID 61.72 noiseFID 60.89 curriculum

#10 1e-4-0.1  4e-5-0.04, 1.5, r0.65, imageFID 56.63 noiseFID 55.78
#102 1e-4-0.1 3e-5-0.03, 1.5, r0.55, imageFID 57.72 noiseFID 56.27
#103 1e-4-0.1 3e-5-0.03, 1.5, -pi-pi, r0.50, imageFID 48.14 noiseFID 48.04, 0.9pi: 35.17/47.77, 29.96/38.07
#104 1e-4-0.1 3e-5-0.03, 1.5, -pi-pi, 100steps, r0.50, imageFID 53.42, noiseFID 51.50 curriculum

#105: 103+300steps 1e-4-0.07, 3e-5-0.02, 29.11/25.83, r0.50
#106: 1000steps 1e-4-0.015,3e-5-0.0045 r0.50

if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
