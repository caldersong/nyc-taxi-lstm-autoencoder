import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, num_epochs=20, batch_size=64, learning_rate=1e-3, device="cpu"):
    """
    Train an LSTM autoencoder on X_train.

    Parameters:
        model: the LSTMAutoencoder model
        X_train: Numpy array of shape (num_samples, seq_len, 1)
        device: "cuda" or "cpu"

    Returns:
        trained model and training loss list 
    """

    model = model.to(device)
    model.train()

    X_tensor = torch.tensor(X_train, dtype=torch.float32) # NumPy array into PyTorch tensor
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # gives minibatch at a time

    criterion = nn.MSELoss() 
    print("Criterion object:", criterion)
    print("Criterion class:", type(criterion))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # adjust model weights to reduce loss

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            print(f"Batch: {i}")
            x = batch[0].to(device)

            # Forard pass
            output = model(x) # produced by decoder 
            loss = criterion(output, x) # reconstruction vs input 

            # Backward & optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() # accumulate loss for this epoch 

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return model, losses 

def train_forecasting_model(model, X_train, y_train, num_epochs=20, batch_size=64, learning_rate=1e-3, device="cpu"):
    """
    Train LSTM forecasting model using input-output pairs (X → y).
    """
    model = model.to(device)
    model.train()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)             # predicted sequence
            loss = criterion(output, y_batch)   # compare to actual future

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return model, losses
