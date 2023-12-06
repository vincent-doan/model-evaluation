import logging
import torch
from torch import nn, optim

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, name):
        super(MLPModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)
        self.name = name

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, log_file, num_epochs=100, learning_rate=0.01):
    logging.basicConfig(filename=log_file, level=logging.INFO)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # ---------- TRAINING ---------- #
        train_loss = 0
        for x_train, y_train in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            train_loss = criterion(outputs, y_train)
            train_loss.backward()
            optimizer.step()

            train_loss += train_loss.item()

        # ---------- VALIDATION ---------- #
        with torch.no_grad():
            val_loss = 0
            for x_val, y_val in val_loader:
                model.eval()
                outputs = model(x_val)
                val_loss = criterion(outputs, y_val)

                val_loss += val_loss.item()

        # ---------- LOGGING ---------- #
        if (epoch + 1) % 10 == 0:
            logging.info(f' Model: {model.name} - Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')
