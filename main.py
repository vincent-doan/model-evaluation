from math import floor, ceil

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split, DataLoader

from one_dim_dataset import UniformDataset, GMMDataset
from models import MLPModel, train_model

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

data = UniformDataset(a=-10, b=10)
data.plot_data_scatter_with_ground_truth(axes[0, 0])
data.plot_x_distribution(axes[0, 1])

data = GMMDataset(means=np.random.uniform(-10, 10, size=10), stds=np.random.uniform(0, 4, size=10))
data.plot_data_scatter_with_ground_truth(axes[1, 0])
data.plot_x_distribution(axes[1, 1])

plt.tight_layout()
plt.savefig('data_distribution.png')

# ------------------------------ HYPERPARAMS ------------------------------ #
INPUT_SIZE = 1
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 0.01
LOG_FILE = 'model_training.log'
SEED = 1989

# ------------------------------ DATA ------------------------------ #
uniform_data = UniformDataset(a=-10, b=10, seed=SEED)
uniform_data_train, uniform_data_val = random_split(uniform_data, [8000, 2000])

gmm_data = GMMDataset(means=np.random.uniform(-10, 10, size=10), stds=np.random.uniform(0, 4, size=10), seed=SEED)
gmm_data_train, gmm_data_val = random_split(gmm_data, [8000, 2000])

uniform_data_train_loader = DataLoader(uniform_data_train, batch_size=32, shuffle=True)
uniform_data_val_loader = DataLoader(uniform_data_val, batch_size=32, shuffle=False)

gmm_data_train_loader = DataLoader(gmm_data_train, batch_size=32, shuffle=True)
gmm_data_val_loader = DataLoader(gmm_data_val, batch_size=32, shuffle=False)

xlim_low = floor(max(uniform_data.min_x, gmm_data.min_x))
xlim_high = ceil(min(uniform_data.max_x, gmm_data.max_x))
ylim_low = floor(max(uniform_data.min_y, gmm_data.min_y))
ylim_high = ceil(min(uniform_data.max_y, gmm_data.max_y))

# ------------------------------ TRAINING ------------------------------ #
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def train_and_plot(model, train_loader, val_loader, data, ax):
    train_model(model, train_loader, val_loader, LOG_FILE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    x_range = np.linspace(floor(data.min_x), ceil(data.max_x), 1000)
    x_range_tensor = torch.tensor(x_range, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        predictions_range = model(x_range_tensor).numpy()

    data.plot_data_scatter_with_ground_truth(ax)
    ax.plot(x_range, predictions_range, color='black', label='Learned function')
    ax.set_title(model.name)
    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(ylim_low, ylim_high)
    ax.legend()

# ------------------------------ Uniform Data ------------------------------ #
uniform_model_1 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=1, name='uniform_model_1')
uniform_model_3 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=3, name='uniform_model_3')
uniform_model_5 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=5, name='uniform_model_5')

train_and_plot(uniform_model_1, uniform_data_train_loader, uniform_data_val_loader, uniform_data, axes[0, 0])
train_and_plot(uniform_model_3, uniform_data_train_loader, uniform_data_val_loader, uniform_data, axes[0, 1])
train_and_plot(uniform_model_5, uniform_data_train_loader, uniform_data_val_loader, uniform_data, axes[0, 2])

# ------------------------------ GMM Data ------------------------------ #
gmm_model_1 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=1, name='gmm_model_1')
gmm_model_3 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=3, name='gmm_model_3')
gmm_model_5 = MLPModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, num_hidden_layers=5, name='gmm_model_5')

train_and_plot(gmm_model_1, gmm_data_train_loader, gmm_data_val_loader, gmm_data, axes[1, 0])
train_and_plot(gmm_model_3, gmm_data_train_loader, gmm_data_val_loader, gmm_data, axes[1, 1])
train_and_plot(gmm_model_5, gmm_data_train_loader, gmm_data_val_loader, gmm_data, axes[1, 2])

# ------------------------------ Plot ------------------------------ #
plt.tight_layout()
plt.savefig('model_predictions.png')

plt.show()