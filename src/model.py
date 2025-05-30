import torch 
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__() # calling nn.Module

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded 
    
    
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, forecast_horizon=6, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        """
        x: input sequence of shape (batch_size, input_len, input_dim)
        returns: predicted sequence of shape (batch_size, forecast_horizon, input_dim)
        """
        batch_size = x.size(0)

        # Encode the input sequence into hidden state 
        _, (hidden, cell) = self.encoder(x)

        # Prepare decoder input (repeat last hidden state)
        decoder_input = torch.zeros((batch_size, 1, self.hidden_dim), device=x.device)

        # Decoding loop 
        outputs = []
        for _ in range(self.forecast_horizon):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.output_layer(out)  
            outputs.append(pred)
            decoder_input = out  # use last output as next input

        return torch.cat(outputs, dim=1)