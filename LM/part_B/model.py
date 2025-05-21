import torch.nn as nn

# Define variational dropout
class VariationalDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)

        return x * mask

# Define the architecture of the model (LSTM)
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, dropout, weight_tying, var_dropout, pad_index, n_layers=1):
        super().__init__()
        # Define the network's layers
        self.use_weight_tying = weight_tying
        self.use_var_drop = var_dropout
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if self.use_var_drop:
            self.emb_dropout = VariationalDropout(dropout)
            self.output_dropout = VariationalDropout(dropout)
        
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        if self.use_weight_tying:
            if emb_size == hidden_size:
                self.proj = None
                self.output = nn.Linear(hidden_size, output_size)
                self.output.weight = self.embedding.weight
            else: # Adjust to match sizes trough an additional projection layer
                self.proj = nn.Linear(hidden_size, emb_size)
                self.output = nn.Linear(emb_size, output_size)
                self.output.weight = self.embedding.weight
        else:
            self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        if self.use_var_drop:
            emb = self.emb_dropout(emb)

        lstm_out, _ = self.lstm(emb)

        if self.use_var_drop:
            lstm_out = self.output_dropout(lstm_out)

        if self.use_weight_tying and self.proj is not None:
            lstm_out = self.proj(lstm_out)

        return self.output(lstm_out).permute(0, 2, 1) 