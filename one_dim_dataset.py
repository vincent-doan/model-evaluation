from math import floor, ceil

import numpy as np
from scipy.stats import gaussian_kde

import torch
from torch.utils.data import Dataset, DataLoader

COMPLICATED_FUNCTION = lambda x: np.sin(x) * np.cos(2*x) + np.exp(-0.1*x) * np.log(np.abs(x) + 1)

class OneDimDataset(Dataset):
    def __init__(self, x:np.ndarray, y:np.ndarray):
        self.x = x
        self.y = y
        self.min_x = self.x.min()
        self.max_x = self.x.max()
        self.min_y = self.y.min()
        self.max_y = self.y.max()

    def __getitem__(self, idx):
        x = torch.Tensor([self.x[idx]])
        y = torch.Tensor([self.y[idx]])
        return x, y

    def __len__(self):
        return len(self.x)
    
    def plot_data_scatter_with_ground_truth(self, ax):
        x_continuous = np.linspace(self.x.min(), self.x.max(), 1000)
        y_continuous = COMPLICATED_FUNCTION(x_continuous)

        ax.plot(x_continuous, y_continuous, label='Ground Truth', color='blue', linestyle='dashed', linewidth=2)
        ax.scatter(self.x, self.y, label='Sampled Data', color='red', alpha=0.008)
        ax.set_title('Complicated Function and Sampled Data')
        ax.set_xlim(floor(self.x.min()), ceil(self.x.max()))
        ax.set_ylim(floor(self.y.min()), ceil(self.y.max()))
        ax.legend()

class UniformDataset(OneDimDataset):
    def __init__(self, a:int, b:int, num_samples:int=10000, seed:int=42):
        self.a = a
        self.b = b
        self.num_samples = num_samples
        self.seed = seed
        
        np.random.seed(seed)
        uniform_x = np.random.uniform(a, b, num_samples)
        self.x = uniform_x
        self.y = COMPLICATED_FUNCTION(uniform_x)
        super().__init__(self.x, self.y)
    
    def plot_x_distribution(self, ax):
        ax.hist(self.x, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black', label='Sampled Data')
        
        uniform_pdf = 1/(self.b-self.a)
        x_range = np.linspace(self.a, self.b, 1000)
        ax.plot(x_range, np.full_like(x_range, uniform_pdf), label='Uniform PDF', color='blue', linestyle='dashed', linewidth=2)

        ax.set_title('Uniform Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()

class GMMDataset(OneDimDataset):
    def __init__(self, means:np.ndarray, stds:np.ndarray, num_samples:int=10000, seed:int=42):
        self.means = means
        self.stds = stds
        self.num_samples = num_samples
        self.seed = seed

        def generate_gmm(means, stds, num_samples):
            if len(means) != len(stds):
                raise ValueError("Number of means and standard deviations must be the same.")

            num_components = len(means)

            samples = []
            for i in range(num_components):
                component_samples = np.random.normal(loc=means[i], scale=stds[i], size=num_samples // num_components)
                samples.append(component_samples)

            samples = np.concatenate(samples)
            return samples

        def create_gmm_pdf(samples):
            pdf = gaussian_kde(samples)
            return pdf

        def sample_from_pdf(pdf, num_samples):
            samples = pdf.resample(size=num_samples)[0]
            return samples

        np.random.seed(seed)
        generated_samples = generate_gmm(means, stds, num_samples)
        self.pdf = create_gmm_pdf(generated_samples)
        sampled_samples = sample_from_pdf(self.pdf, num_samples)

        self.x = sampled_samples
        self.y = COMPLICATED_FUNCTION(sampled_samples)
        super().__init__(self.x, self.y)

    def plot_x_distribution(self, ax):
        ax.hist(self.x, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black', label='Sampled Data')

        x_range = np.linspace(min(self.x), max(self.x), 1000)
        ax.plot(x_range, self.pdf(x_range), label='PDF', color='blue', linestyle='dashed', linewidth=2)

        ax.set_title('GMM Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()