import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from two_dim import two_dim_ds
import numpy
from torch.utils.data.distributed import DistributedSampler
from dataset import *
from torch.utils.data import random_split

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, one_channel=False, cmap='jet',**kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)  # [C, H, W]
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()        # [H, W, C]
    if one_channel:
        # Squeeze to [H, W]
        img_2d = ndarr[:, :, 0]
        # Apply colormap directly (assuming img_2d in [0, 255])
        img_2d = img_2d.astype(numpy.uint8)
        colormapped = plt.get_cmap(cmap)(img_2d / 255.0)   # colormap expects [0, 1]
        rgb_img = (colormapped[:, :, :3] * 255).astype(numpy.uint8)  # Drop alpha
        im = Image.fromarray(rgb_img, mode='RGB')
    else:
        im = Image.fromarray(ndarr.astype(numpy.uint8))

    im.save(path)

    im.save(path)
def save_two_dim(data, path, range=20):
    data_np = data.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(data_np[:, 0], data_np[:, 1], c='blue', alpha=0.7, edgecolors='k', s=50)
    plt.title('2D Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-range, range)
    plt.ylim(-range, range)
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def save_one_dim(data, path, range=50):
    """
    Save a 1D scatter plot of the data.

    Args:
      data (torch.Tensor): 1D tensor containing values to plot.
      path (str): Path to save the plot.
      range (int): Range of the x-axis.
    """
    data_np = data.reshape(-1).detach().cpu().numpy()  # Convert to numpy

    hist, bin_edges = numpy.histogram(data_np, bins=50, range=[-range, range], density=True)

    # 3. Compute bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 4. Plot
    plt.figure(figsize=(8, 3))
    plt.plot(bin_centers, hist, color='blue', linewidth=2)
    plt.title("Density Map")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.ylim((0.,1.))
    plt.grid(True, linestyle='--', alpha=0.7)

    # 5. Save and close
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_circular_density_plot(data, path, bins=100):
    """
    Save a density visualization of data points mapped to a unit circle.

    Args:
      data (torch.Tensor): 1D tensor containing angles in radians.
      path (str): Path to save the plot.
      bins (int): Number of bins for the density plot.
    """
    data_np = data.detach().cpu().numpy()  # Convert to numpy

    # Convert angles to (x, y) coordinates on a unit circle
    x = numpy.cos(data_np)
    y = numpy.sin(data_np)

    plt.figure(figsize=(6, 6))

    # Hexbin Plot: Density Estimation
    hb = plt.hexbin(x, y, gridsize=bins, cmap='Blues', mincnt=1, linewidths=0.5, alpha=0.75)
    plt.colorbar(hb, label="Density")  # Add colorbar

    # Draw a unit circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed', linewidth=1.5)
    plt.gca().add_patch(circle)

    # Set limits & aspect ratio
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')  # Ensure a perfect circle

    # Labels & grid
    plt.xlabel('Cos(θ)')
    plt.ylabel('Sin(θ)')
    plt.title('Circular Density Plot')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save and close figure
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def get_data(args):
    #dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    if args.dataset_name == "cifar":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root = args.dataset_path, download=True, transform=transforms, train=True)
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, download=True, transform=transforms, train=False)
    elif args.dataset_name == "ns":
        dataset = LMDBData(root=os.path.join(args.dataset_path,"navier-stokes-train/Re200.0-t5.0"),resolution=128,std=5.0)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        #test_dataset = LMDBData(root=os.path.join(args.dataset_path, "navier-stokes-val/Re200.0-t5.0"), resolution=128, std=5.0)
    elif args.dataset_name == "fp":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((96,96)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        print(os.path.join(args.dataset_path,"socofing/socofing/SOCOFing/Real"))
        dataset = BMPDataset(root=os.path.join(args.dataset_path,"socofing/socofing/SOCOFing/Real"),transform=transforms)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, _ = random_split(dataset, [train_size, test_size])
        test_dataset =dataset
    elif args.dataset_name =='tex':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((32,32)),
            #torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
        #train_dataset = test_dataset = dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def get_two_dim(args):
    dataset = two_dim_ds(npar=10000,data_tag='circle')
    train_size = int(0.8 * len(dataset))  # 70% for training
    test_size = int(0.2 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("noise", exist_ok=True)
    os.makedirs("phase", exist_ok=True)
    os.makedirs("recon_images", exist_ok=True)
    os.makedirs("recon_phases", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("noise", run_name), exist_ok=True)
    os.makedirs(os.path.join("phase", run_name), exist_ok=True)
    os.makedirs(os.path.join("recon_images", run_name), exist_ok=True)
    os.makedirs(os.path.join("recon_phases", run_name), exist_ok=True)

