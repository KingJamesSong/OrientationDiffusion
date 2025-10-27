import os
import copy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fid_evaluation import FIDEvaluation
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, EMA, UNet2PhaseAmp, UNet2AmpPhase
import logging
import numpy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from scipy.optimize import curve_fit
from numpy.linalg import LinAlgError

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
    phase = torch.atan2(x[:,1:2], x[:,0:1])
    # Compute edge magnitude
    mag = torch.sqrt(x[:,0:1]**2 + x[:,1:2]**2) + 1e-6
    return mag, phase

def phase_modulate(x):
    x = (x + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return x

def save_phase(phase):
    phase = phase / numpy.pi
    phase = (phase + 1) / 2.
    phase = (phase * 255).type(torch.uint8)
    return phase

def save_mag(mag):
    B = mag.shape[0]
    mags = []
    for i in range(B):
        m = mag[i].log()
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        m = (m * 255).type(torch.uint8)
        mags.append(m)
    return torch.stack(mags)


def map_to_image(mag, phase):
    x = mag * phase.cos()
    y = mag * phase.sin()
    xy =  torch.cat([x,y],dim=1)
    return xy

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

def radial_energy_spectrum(vel) :
    """
    Compute 1D radial energy spectrum from 2D velocity field.
    Args:
        vel: Tensor of shape [2, H, W] (vx, vy)
    Returns:
        spectrum: np.ndarray of shape [H//2] averaged over angle
    """
    vx, vy = vel[0], vel[1]
    H, W = vx.shape
    v_complex = vx + 1j * vy
    v_fft = torch.fft.fft2(v_complex)
    v_fft = torch.fft.fftshift(v_fft)
    energy = torch.abs(v_fft) ** 2

    ky = torch.fft.fftshift(torch.fft.fftfreq(H)) * H
    kx = torch.fft.fftshift(torch.fft.fftfreq(W)) * W
    KX, KY = torch.meshgrid(kx, ky, indexing='xy')
    k_radius = torch.sqrt(KX ** 2 + KY ** 2).cpu().numpy().flatten()
    energy_flat = energy.cpu().numpy().flatten()

    k_bins = numpy.arange(0.5, H // 2 + 1, 1.0)
    k_bin_idx = numpy.digitize(k_radius, k_bins)
    spectrum = numpy.zeros(len(k_bins) + 1)
    counts = numpy.zeros(len(k_bins) + 1)

    for i in range(len(k_radius)):
        bin_idx = k_bin_idx[i]
        spectrum[bin_idx] += energy_flat[i]
        counts[bin_idx] += 1

    counts[counts == 0] = 1
    return (spectrum / counts)[1:H // 2]

def fit_slope_auto(spectrum, H, W, frac_range=(0.1, 0.4)):
    """
    Automatically fit slope of log-log energy spectrum in a dynamic wavenumber range.

    Args:
        spectrum: 1D numpy array of E(k)
        H, W: spatial resolution of the velocity field
        frac_range: tuple, fraction of [1, H//2] to define fit range (default 10%–40%)

    Returns:
        slope: fitted slope
        log_fit: full fitted curve for plotting
        fit_range: (k_start, k_end) used for fitting
    """
    eps = 1e-20  # avoid log(0)
    spectrum = numpy.clip(spectrum, eps, None)
    k = numpy.arange(1, len(spectrum) + 1)
    log_k = numpy.log(k)
    log_E = numpy.log(spectrum + 1e-8)

    k_max = min(H, W) // 2
    k_start = int(k_max * frac_range[0])
    k_end = int(k_max * frac_range[1])
    mask = (k >= k_start) & (k <= k_end)

    log_k_fit = log_k[mask]
    log_E_fit = log_E[mask]
    try:
        slope, intercept = numpy.polyfit(log_k_fit, log_E_fit, deg=1)
    except LinAlgError:  # rare SVD failure
        A = numpy.vstack([log_k_fit, numpy.ones_like(log_k_fit)]).T
        slope, intercept = numpy.linalg.lstsq(A, log_E_fit, rcond=None)[0]

    log_fit = slope * log_k + intercept
    return slope, log_fit, (k_start, k_end)

def log_mse_distance(spec1, spec2):
    """Compute log-MSE between two spectra."""
    log_spec1 = numpy.log(spec1 + 1e-8)
    log_spec2 = numpy.log(spec2 + 1e-8)
    return numpy.mean((log_spec1 - log_spec2) ** 2)

def spectrum_metrics_batch(real, fake):
    """
    Compute slope difference and Wasserstein distance between spectra of real and fake velocity fields.

    Args:
        real, fake: torch.Tensor of shape [B, 2, H, W]

    Returns:
        slope_diff: abs diff between spectral slopes
        wass_dist: Wasserstein distance between mean energy spectra
        mean_real: numpy array, mean real spectrum
        mean_fake: numpy array, mean fake spectrum
        slope_real: float
        slope_fake: float
        fit_real: numpy array, fitted log spectrum curve
        fit_fake: numpy array
    """
    # Compute radial spectra for all samples
    real_spec = numpy.stack([radial_energy_spectrum(real[i]) for i in range(real.size(0))])
    fake_spec = numpy.stack([radial_energy_spectrum(fake[i]) for i in range(fake.size(0))])

    # Normalize each spectrum to sum to 1 (for Wasserstein comparison)
    real_spec_norm = real_spec / (real_spec.sum(axis=1, keepdims=True) + 1e-8)
    fake_spec_norm = fake_spec / (fake_spec.sum(axis=1, keepdims=True) + 1e-8)

    # Mean spectrum
    mean_real = real_spec_norm.mean(axis=0)
    mean_fake = fake_spec_norm.mean(axis=0)

    # Fit slopes on log-log
    slope_real, fit_real, fit_range = fit_slope_auto(mean_real, real.size(2), real.size(3))
    slope_fake, fit_fake, _ = fit_slope_auto(mean_fake, fake.size(2), fake.size(3))

    # Metric computations
    slope_diff = abs(slope_real - slope_fake)
    wass_dist = wasserstein_distance(mean_real, mean_fake)
    log_mse = log_mse_distance(mean_real, mean_fake)

    return log_mse, slope_diff, wass_dist, mean_real, mean_fake, slope_real, slope_fake, fit_real, fit_fake, fit_range


def angle_space(score, angle):
    C = angle.size(1)
    sin_score = score[:,:C,:,:]
    cos_score = score[:,C:,:,:]
    score = -sin_score * torch.sin(angle) + cos_score * torch.cos(angle)
    return score

#Two Diffusion Models (Kuramoto Phase + VP-SDE Log Magnitude)
class JointDiffusion:
    def __init__(self, noise_steps=100, noise_start=1e-4, noise_end=0.1, coupling_start=3e-3, coupling_end=0.045, beta_start=1e-4, beta_end=0.15, img_size=32, cutoff=6, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        #Orientation
        self.coupling_st = torch.linspace(coupling_start, coupling_end, self.noise_steps).to(self.device)
        self.noise_st = self.prepare_noise_schedule(noise_start, noise_end)
        self.ref_phase = 0.#numpy.pi
        #Amplitude
        self.beta = torch.linspace(beta_start, beta_end, self.noise_steps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_hat[:-1]], dim=0)
        self.cutoff = cutoff

    def prepare_noise_schedule(self, noise_start, noise_end):
        return torch.linspace(noise_start, noise_end, self.noise_steps).sqrt().to(self.device)

    def coupling_noise(self, x):
        w = torch.randn_like(x)
        complex_phases = torch.complex(x.cos(),x.sin())
        mean_phase_vector = torch.mean(complex_phases,dim=[1,2,3],keepdim=True)
        order_phase = torch.atan2(mean_phase_vector.imag, mean_phase_vector.real)
        order_parameter = torch.sqrt(mean_phase_vector.imag**2 + mean_phase_vector.real**2)
        deltax = order_parameter * torch.sin(order_phase - x) + 1.5*torch.sin(self.ref_phase - x)
        return w, deltax, order_parameter, order_phase

    def coupling_noise_local(self, x, M=4):
        #B, C, H, W = x.size()
        w = torch.randn_like(x)
        complex_phases = torch.exp(1j * x)
        kernel = torch.ones(1, 1, 2 * M + 1, 2 * M + 1) / ((2 * M + 1) ** 2)
        kernel = kernel.repeat(1,1,1,1).to(x)
        # Apply circular padding to handle periodic boundary conditions
        complex_padded = F.pad(complex_phases, (M, M, M, M), mode='circular')
        order_phase = torch.atan2(complex_padded.imag, complex_padded.real)
        order_parameter = torch.sqrt(complex_padded.imag ** 2 + complex_padded.real ** 2)
        order_parameter = F.conv2d(order_parameter, kernel, groups=1)
        order_phase = F.conv2d(order_phase, kernel, groups=1)
        deltax1 = order_parameter * torch.sin(order_phase - x)
        deltax2 = torch.sin(self.ref_phase - x)
        return w, deltax1, deltax2, order_parameter*torch.cos(order_phase-x), order_phase

    def amplitude_prefactor_local(self, r, M=4):
        # normalized box kernel for mean over N_i
        k = 2 * M + 1
        kernel = torch.ones((1, 1, k, k), device=r.device, dtype=r.dtype) / (k * k)
        # periodic (or chosen) padding, then depth-1 conv
        r_pad = F.pad(r, (M, M, M, M), mode='circular')
        r_mean = F.conv2d(r_pad, kernel)  # [B,1,H,W], local mean of r^alpha
        S = r * r_mean
        return S

    def noise_amp(self, x, t, acc_coherence):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + acc_coherence + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def noise_images(self, x, log_amp_0, t):
        acc_coherence = torch.zeros_like(log_amp_0)
        for i in range(0, t.max()):
            beta = self.beta[i]
            log_amp_i, _ = self.noise_amp(log_amp_0, (torch.ones(x.size(0)) * i).long().to(self.device), acc_coherence)
            mask1 = ((i > self.noise_steps//self.cutoff) & (t > i)).nonzero().squeeze(dim=1)
            mask2 = ((i <= self.noise_steps//self.cutoff) & (t > i)).nonzero().squeeze(dim=1)

            w, deltax1, deltax2, _, _ = self.coupling_noise_local(x[mask1])
            x[mask1] = x[mask1] + self.coupling_st[i] * (deltax1 + deltax2) + self.noise_st[i] * w
            x[mask1] = phase_modulate(x[mask1])

            amp_factor = log_amp_i.exp()[mask2]
            w, deltax1, deltax2, order_parameter, _ = self.coupling_noise_local(x[mask2])
            x[mask2] = x[mask2] + self.coupling_st[i] * amp_factor * (deltax1 + deltax2) + self.noise_st[i] * w
            x[mask2] = phase_modulate(x[mask2])
            acc_coherence[mask2] += torch.sqrt(self.alpha_hat[t][mask2][:, None, None, None] / self.alpha_prev[i]) * 0.5 * beta * order_parameter

        log_amp_t, _ = self.noise_amp(log_amp_0, t, acc_coherence)
        beta_t = self.beta[t][:, None, None, None]
        mask = ((t+1) <= self.noise_steps//self.cutoff).nonzero().squeeze(dim=1)
        _, _, _, order_parameter, _ = self.coupling_noise_local(x[mask])
        acc_coherence[mask] += torch.sqrt(self.alpha_hat[t + 1][mask][:, None, None, None] / self.alpha_prev[t][mask][:, None, None, None]) * 0.5 * beta_t[mask] * order_parameter
        log_amp_t1, _ = self.noise_amp(log_amp_0, t+1, acc_coherence)

        return x, log_amp_t, log_amp_t1, acc_coherence


    def compute_score(self, x_t, x_0, t, acc_coherence):
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        coef = 1. / (1. - alpha_hat_t)
        score = -coef * (x_t - torch.sqrt(alpha_hat_t) * x_0 - acc_coherence)
        return score, 1. - alpha_hat_t

    def sample_timesteps(self,n, epoch):      
        return torch.randint(low=0, high=self.noise_steps-1, size=(n,))

    def sample_image(self, model1, x, amp, model2):
        model1.eval()
        model2.eval()
        with torch.no_grad():
            log_amp_0 = torch.log(torch.clamp(amp, min=1e-6))
            acc_coherence = torch.zeros_like(log_amp_0)
            for i in tqdm(range(0, self.noise_steps-1), position=0):
                t = (torch.ones(x.size(0)) * i).long().to(self.device)
                log_amp_i, _ = self.noise_amp(log_amp_0, t, acc_coherence)
                w, deltax1, deltax2, order_parameter, order_phase = self.coupling_noise_local(x)
                if i > self.noise_steps//self.cutoff:
                    x = x + self.coupling_st[i] * (deltax1 + deltax2) + self.noise_st[i] * w
                else:
                    amp_factor = log_amp_i.exp()
                    x = x + self.coupling_st[i] * amp_factor * (deltax1 + deltax2) + self.noise_st[i] * w
                    acc_coherence += torch.sqrt(self.alpha_hat[int(self.noise_steps//self.cutoff)]/self.alpha_prev[i]) * 0.5 * self.beta[i] * order_parameter
                x = phase_modulate(x)
            log_amp, _ = self.noise_amp(log_amp_0, (torch.ones(x.size(0)) * i).long().to(self.device), acc_coherence)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # phase diffusion
                t = (torch.ones(1) * i).long().to(self.device)
                w, deltax1, deltax2, order_parameter, order_phase = self.coupling_noise_local(x)
                score = model1(x, log_amp.exp(), t)
                score = angle_space(score, x)
                if i > self.noise_steps//self.cutoff:
                    x = x - self.coupling_st[i] * (deltax1 + deltax2) + (self.noise_st[i] ** 2) * score + self.noise_st[i] * w
                else:
                    amp_factor = log_amp.exp()
                    x = x - self.coupling_st[i] * amp_factor * (deltax1 + deltax2) + (self.noise_st[i] ** 2) * score + self.noise_st[i] * w
                x = phase_modulate(x)
                # magnitude diffusion
                beta = self.beta[t][:, None, None, None]
                sqrt_beta = torch.sqrt(beta)
                score = model2(log_amp, order_parameter, t)  # sθ(x, t)
                # Euler–Maruyama update:
                if i > self.noise_steps//self.cutoff:
                    drift = 0.5 * beta * log_amp + beta * score
                else:
                    drift = 0.5 * beta * log_amp - 0.5 * beta * order_parameter + beta * score
                noise = sqrt_beta * torch.randn_like(log_amp)
                log_amp = log_amp + drift + noise  # reverse SDE step
        model1.train()
        model2.train()
        return x, log_amp.exp()

def train_joint(args):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, test_dataloader = get_data(args)
    model1 = UNet2PhaseAmp(c_in=1, c_out=1,img_size=args.image_size).to(device)
    model2 = UNet2AmpPhase(c_in=1, c_out=1, img_size=args.image_size).to(device)
    #model1.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt1.pt")))
    #model2.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt2.pt")))
    optimizer = optim.AdamW([{'params': model1.parameters()}, {'params': model2.parameters()}]
                            , lr=args.lr)
    mse = nn.MSELoss()
    diffusion = JointDiffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_dataloader)
    ema = EMA(0.995)
    ema_model1 = copy.deepcopy(model1).eval().requires_grad_(False)
    ema_model2 = copy.deepcopy(model2).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            mag, phase = map_to_phase(images)
            t = diffusion.sample_timesteps(mag.size(0), epoch).to(device)
            #OrientationDiffusion
            loss = 0.0
            exp_sam = 5
            log_mag = torch.log(torch.clamp(mag, min=1e-6))
            phase_t, log_mag_t, log_mag_t1, acc_coherence  = diffusion.noise_images(phase, log_mag, t)
            score2, var = diffusion.compute_score(log_mag_t1, log_mag, t + 1, acc_coherence)
            mask = ((t + 1) > diffusion.noise_steps//diffusion.cutoff).nonzero().squeeze(dim=1)
            mag_t1 = log_mag_t1.exp()
            amp_factor = log_mag_t.exp()
            for sample2 in range(exp_sam):
                w, deltax1, deltax2, _, _ = diffusion.coupling_noise_local(phase_t)
                Fxt = phase_t + diffusion.coupling_st[t][:, None, None, None] * amp_factor * (deltax1 + deltax2)
                Fxt[mask] = Fxt[mask] + diffusion.coupling_st[t][mask][:, None, None, None] * (1 - amp_factor[mask]) * (deltax1[mask] + deltax2[mask])
                Fxt = phase_modulate(Fxt)
                x_t1 = Fxt + diffusion.noise_st[t][:, None, None, None] * w
                x_t1 = phase_modulate(x_t1)
                phase_score = model1(x_t1, mag_t1 , t + 1)
                phase_score = angle_space(phase_score, x_t1)
                score = wrapped_gaussian_score(x_t1, Fxt, diffusion.noise_st[t][:, None, None, None])
                loss += mse(phase_score * diffusion.noise_st[t + 1][:, None, None, None] ** 2 / 2,
                            score * diffusion.noise_st[t + 1][:, None, None, None] ** 2 / 2)
            loss = loss / exp_sam
            #Amplitude Diffusion
            _, _, _, order_parameter, _ = diffusion.coupling_noise_local(x_t1)
            amp_score = model2(log_mag_t1, order_parameter, t+1)
            loss += mse(amp_score * var, score2 * var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model1, model1)
            ema.step_ema(ema_model2, model2)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        if epoch%5==0:
            sampled_images_phase, sampled_images_mag = diffusion.sample_image(ema_model1, phase, mag, ema_model2)
            recon = map_to_image(sampled_images_mag, sampled_images_phase)
            save_quiver_grid_image(recon, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(save_phase(sampled_images_phase), os.path.join("results", args.run_name, f"phase_{epoch}.jpg"), one_channel=True,cmap='viridis')
            save_images(save_mag(sampled_images_mag), os.path.join("results", args.run_name, f"mag_{epoch}.jpg"), one_channel=True, cmap='viridis')
        torch.save(model1.state_dict(), os.path.join("models", args.run_name, f"ckpt1.pt"))
        torch.save(model2.state_dict(), os.path.join("models", args.run_name, f"ckpt2.pt"))
        torch.save(ema_model1.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt1.pt"))
        torch.save(ema_model2.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt2.pt"))

def validate(args):
    device = args.device
    setup_logging(args.run_name)
    args.batch_size = 40
    train_dataloader, test_dataloader = get_data(args)
    model1 = UNet2PhaseAmp(c_in=1, c_out=1, img_size=args.image_size).to(device)
    model2 = UNet2AmpPhase(c_in=1, c_out=1, img_size=args.image_size).to(device)
    #load pre-trained model
    model1.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt1.pt")))
    model2.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt2.pt")))
    diffusion = JointDiffusion(img_size=args.image_size, device=device)
    pbar = tqdm(test_dataloader)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for sample, (images, _) in enumerate(pbar):
            if sample>0:
                break
            images = images.to(device)
            #map to phase space
            mag, phases = map_to_phase(images)
            temp = (images - images.min()) / (images.max() - images.min() + 1e-6)  # map to [0, 1]
            save_images(temp[:,0:1]*255, os.path.join("noise", args.run_name, "-1_x.jpg"),one_channel=True,cmap = 'viridis')
            save_images(temp[:,1:2]*255, os.path.join("noise", args.run_name, "-1_y.jpg"), one_channel=True,cmap = 'viridis')
            save_images(save_phase(phases), os.path.join("phase", args.run_name, "-1.jpg"),one_channel=True,cmap = 'viridis')
            save_images(save_mag(mag), os.path.join("mag", args.run_name, "-1.jpg"),one_channel=True,cmap = 'viridis')
            save_quiver_grid_image(images, os.path.join("noise", args.run_name, "-1_xy.jpg"))
            for i in tqdm(range(0, diffusion.noise_steps-1), position=0):
                log_amp_0 = torch.log(torch.clamp(mag, min=1e-6))
                acc_coherence = torch.zeros_like(log_amp_0)
                t = (torch.ones(phases.size(0)) * i).long().to(diffusion.device)
                log_amp_i, _ = diffusion.noise_amp(log_amp_0, t, acc_coherence)
                w, deltax1, deltax2, order_parameter, order_phase = diffusion.coupling_noise_local(phases)
                if i > diffusion.noise_steps // diffusion.cutoff:
                    phases = phases + diffusion.coupling_st[i] * (deltax1 + deltax2) + diffusion.noise_st[i] * w
                else:
                    amp_factor = log_amp_i.exp()
                    phases = phases + diffusion.coupling_st[i] * amp_factor * (deltax1 + deltax2) + diffusion.noise_st[i] * w
                    acc_coherence += torch.sqrt(
                        diffusion.alpha_hat[int(diffusion.noise_steps // diffusion.cutoff)] / diffusion.alpha_prev[i]) * 0.5 * \
                                     diffusion.beta[i] * order_parameter
                phases = phase_modulate(phases)
                recon = map_to_image(log_amp_i.exp(), phases)
                temp = (recon - recon.min()) / (recon.max() - recon.min() + 1e-6)  # map to [0, 1]
                save_images(temp[:,0:1]*255, os.path.join("noise", args.run_name, f"x_{i}.jpg"),one_channel=True,cmap = 'viridis')
                save_images(temp[:,1:2]*255, os.path.join("noise", args.run_name, f"y_{i}.jpg"),one_channel=True,cmap = 'viridis')
                save_images(save_phase(phases), os.path.join("phase", args.run_name,  f"{i}.jpg"),one_channel=True,cmap = 'viridis')
                save_images(save_mag(log_amp_i.exp()), os.path.join("mag", args.run_name,  f"{i}.jpg"),one_channel=True, cmap = 'viridis')
                save_quiver_grid_image(recon, os.path.join("noise", args.run_name, f"xy_{i}.jpg"))
            log_amp, _ = diffusion.noise_amp(log_amp_0, (torch.ones(phases.size(0)) * i).long().to(diffusion.device), acc_coherence)
            for i in tqdm(reversed(range(1, diffusion.noise_steps)), position=0):
                t = (torch.ones(1) * i).long().to(diffusion.device)
                w, deltax1, deltax2, order_parameter, order_phase = diffusion.coupling_noise_local(phases)
                score = model1(phases, log_amp.exp(), t)
                score = angle_space(score, phases)
                if i > diffusion.noise_steps // diffusion.cutoff:
                    phases = phases - diffusion.coupling_st[i] * (deltax1 + deltax2) + (diffusion.noise_st[i] ** 2) * score + diffusion.noise_st[i] * w
                else:
                    amp_factor = log_amp.exp()
                    phases = phases - diffusion.coupling_st[i] * amp_factor * (deltax1 + deltax2) + (diffusion.noise_st[i] ** 2) * score + \
                        diffusion.noise_st[i] * w
                phases = phase_modulate(phases)
                # magnitude diffusion
                beta = diffusion.beta[t][:, None, None, None]
                sqrt_beta = torch.sqrt(beta)
                score = model2(log_amp, order_parameter, t)  # sθ(x, t)
                # Euler–Maruyama update:
                if i > diffusion.noise_steps // diffusion.cutoff:
                    drift = 0.5 * beta * log_amp + beta * score
                else:
                    drift = 0.5 * beta * log_amp - 0.5 * beta * order_parameter + beta * score
                noise = sqrt_beta * torch.randn_like(log_amp)
                log_amp = log_amp + drift + noise  # reverse SDE step
                recon = map_to_image(log_amp.exp(), phases)
                temp = (recon - recon.min()) / (recon.max() - recon.min() + 1e-6)  # map to [0, 1]
                save_images(temp[:, 0:1] * 255, os.path.join("noise", args.run_name, f"rex_{i}.jpg"), one_channel=True,
                            cmap='viridis')
                save_images(temp[:, 1:2] * 255, os.path.join("noise", args.run_name, f"rey_{i}.jpg"), one_channel=True,
                            cmap='viridis')
                save_images(save_phase(phases), os.path.join("phase", args.run_name, f"re_{i}.jpg"), one_channel=True,
                            cmap='viridis')
                save_images(save_mag(log_amp.exp()), os.path.join("mag", args.run_name, f"re_{i}.jpg"), one_channel=True,
                            cmap='viridis')
                save_quiver_grid_image(recon, os.path.join("noise", args.run_name, f"rexy_{i}.jpg"))
        #text_save(os.path.join("phase", args.run_name, "./all_order.txt"),numpy.array(order.view(-1).numpy()))
        #text_save(os.path.join("phase", args.run_name, "./all_phase.txt"),numpy.array(phase.view(-1).numpy()))

def evaluate_dataloaders_spectral(args, plot=True):
    device = args.device
    setup_logging(args.run_name)
    args.batch_size = 1024
    train_dataloader, test_dataloader = get_data(args)
    model1 = UNet2PhaseAmp(c_in=1, c_out=1, img_size=args.image_size).to(device)
    model2 = UNet2AmpPhase(c_in=1, c_out=1, img_size=args.image_size).to(device)
    # load pre-trained model
    model1.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt1.pt")))
    model2.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt2.pt")))
    #ori_diffusion = OrientationDiffusion(img_size=args.image_size, device=device)
    #std_diffusion = Diffusion(img_size=args.image_size, device=device)
    diffusion = JointDiffusion(img_size=args.image_size, device=device)
    pbar = tqdm(test_dataloader)
    model1.eval()
    model2.eval()
    all_real, all_fake = [], []
    with torch.no_grad():
        for sample, (images, _) in enumerate(pbar):
            images = images.to(device)

            mag, phase = map_to_phase(images)
            sampled_phase, sampled_mag = diffusion.sample_image(model1, phase, mag, model2)
            #sampled_mag = std_diffusion.sample_score_model(model2, mag)
            fake = map_to_image(sampled_mag, sampled_phase)

            all_real.append(images.cpu())
            all_fake.append(fake.cpu())

        real_vel = torch.cat(all_real, dim=0)
        fake_vel = torch.cat(all_fake, dim=0)

        log_mse, slope_diff, wass_dist, real_mean, fake_mean, slope_real, slope_fake, fit_real, fit_fake, fit_range = spectrum_metrics_batch(real_vel, fake_vel)

        if plot:
            k = numpy.arange(1, len(real_mean) + 1)
            k_fit_mask = (k >= fit_range[0]) & (k <= fit_range[1])
            plt.figure()
            plt.loglog(k, real_mean, label='Real', linewidth=2.5)
            plt.loglog(k, fake_mean, label='Fake', linewidth=2.5)
            plt.loglog(k[k_fit_mask], numpy.exp(fit_real)[k_fit_mask], '--', linewidth=1.5, label=f"Real Fit (slope={slope_real:.2f})")
            plt.loglog(k[k_fit_mask], numpy.exp(fit_fake)[k_fit_mask], '--', linewidth=1.5, label=f"Fake Fit (slope={slope_fake:.2f})")
            plt.axvspan(fit_range[0], fit_range[1], color='gray', alpha=0.2, label='Fit Range')
            plt.xlabel('Wavenumber k')
            plt.ylabel('E(k)')
            plt.title('Spectrum Comparison')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join("noise", args.run_name, "hist.jpg"),dpi=300)
            plt.close()

        print(f"[{args.run_name}] Slope Difference: {slope_diff:.4f}")
        print(f"[{args.run_name}] Wasserstein Distance: {wass_dist:.4f}")
        print(f"[{args.run_name}] Log MSE: {log_mse:.4f}")
        return slope_diff, wass_dist, log_mse

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "OrientationNS"
    args.dataset_name = "ns"
    args.epochs = 300
    args.batch_size = 48
    args.image_size = 128
    args.patch_size = 8
    args.dataset_path = "/home/yuesong/pdearena/"
    args.device = "cuda"
    args.lr = 1e-4
    train_joint(args)
    evaluate_dataloaders_spectral(args)
    validate(args)


if __name__ == '__main__':
    launch()
    
