import h5py
import os
import glob
import torch
from torch.utils.data import Dataset
import xarray as xr
import torch.nn.functional as F
import json

class NavierStokesVelocityDataset(Dataset):
    def __init__(self, path, mode="train", normalize=True):
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        self.normalize = normalize
        self.samples = []
        self.handles = []
        self.files = []

        # Collect all .h5 files in the directory
        if os.path.isdir(path):
            h5_files = sorted(glob.glob(os.path.join(path, "*.h5")))
        elif os.path.isfile(path) and path.endswith(".h5"):
            h5_files = [path]
        else:
            raise ValueError(f"Invalid path: {path}")

        # Filter files that contain the requested mode
        for fpath in h5_files:
            with h5py.File(fpath, 'r') as f:
                if mode in f:
                    self.files.append(fpath)

        if not self.files:
            raise FileNotFoundError(f"No files found with mode '{mode}' in: {path}")

        # Build sample index list
        for file_idx, fpath in enumerate(self.files):
            f = h5py.File(fpath, 'r')
            self.handles.append(f)
            grp = f[mode]
            N, T = grp["vx"].shape[:2]
            for n in range(N):
                self.samples.append((file_idx, n, T - 1))  # Use last time step

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, n, t = self.samples[idx]
        f = self.handles[file_idx]
        mode = list(f.keys())[0]
        grp = f[mode]

        vx = torch.tensor(grp["vx"][n, t])
        vy = torch.tensor(grp["vy"][n, t])
        velocity = torch.stack([vx, vy], dim=0).float()

        if self.normalize:
            velocity = (velocity - velocity.mean()) / (velocity.std() + 1e-6)

        filename = os.path.basename(self.files[file_idx])
        identifier = f"{filename}::n={n},t={t}"

        return velocity, identifier

    def close(self):
        for f in self.handles:
            f.close()

class NavierStokesVorticitySeqDataset(Dataset):
    """
    Returns a 3-step vorticity sequence [ω_{t-1}, ω_t, ω_{t+1}] with shape [3, H, W].
    Assumes HDF5 structure: f[mode]["vx"], f[mode]["vy"] with shape [N, T, H, W].
    Vorticity ω = ∂x v_y - ∂y v_x computed with central differences and periodic BC.
    """
    def __init__(self, path, mode="train", normalize=True, dx=1.0, dy=1.0):
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        self.mode = mode
        self.normalize = normalize
        self.dx, self.dy = float(dx), float(dy)
        self.samples, self.handles, self.files = [], [], []

        # Collect .h5 files
        if os.path.isdir(path):
            h5_files = sorted(glob.glob(os.path.join(path, "*.h5")))
        elif os.path.isfile(path) and path.endswith(".h5"):
            h5_files = [path]
        else:
            raise ValueError(f"Invalid path: {path}")

        # Keep files that contain the requested mode
        for fpath in h5_files:
            with h5py.File(fpath, "r") as f:
                if mode in f:
                    self.files.append(fpath)
        if not self.files:
            raise FileNotFoundError(f"No files found with mode '{mode}' in: {path}")

        # Build index: (file_idx, n, t) where sequence is [t-1, t, t+1]
        for file_idx, fpath in enumerate(self.files):
            f = h5py.File(fpath, "r")
            self.handles.append(f)
            grp = f[self.mode]
            N, T = grp["vx"].shape[:2]
            # t runs from 1 .. T-2 so that t-1 and t+1 are valid
            for n in range(N):
                for t in range(1, T - 1):
                    self.samples.append((file_idx, n, t))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        file_idx, n, t = self.samples[idx]
        f = self.handles[file_idx]
        grp = f[self.mode]  # explicit

        # Load 3 consecutive timesteps: t-1, t, t+1
        vx_seq = torch.tensor(grp["vx"][n, t - 1:t + 2])  # [3, H, W]
        vy_seq = torch.tensor(grp["vy"][n, t - 1:t + 2])  # [3, H, W]

        # Compute vorticity per timestep
        w_list = []
        for k in range(3):
            vx = vx_seq[k].to(torch.float32)
            vy = vy_seq[k].to(torch.float32)
            w = torch.stack([vx, vy], dim=0).float() # [H, W]
            w_list.append(w)
        w_seq = torch.stack(w_list, dim=0)  # [3, H, W] = [ω_{t-1}, ω_t, ω_{t+1}]

        # --- make last item the delta: Δω = ω_{t+1} - ω_t ---
        #w_seq[2] = w_seq[2] - w_seq[1]  # now [ω_{t-1}, ω_t, Δω]

        if self.normalize:
            # simple per-sample z-score over the whole 3-channel tensor
            mean = w_seq.mean()
            std = w_seq.std().clamp_min(1e-6)
            w_seq = (w_seq - mean) / std

        filename = os.path.basename(self.files[file_idx])
        identifier = f"{filename}::n={n},tseq={t - 1}-{t + 1}"

        return w_seq, identifier  # [3, H, W] = [ω_{t-1}, ω_t, Δω]

    def close(self):
        for f in self.handles:
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

def safe_open_dataset(path):
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception as e:
        print(f"Warning: Skipping corrupted file: {path} — {e}")
        return None
