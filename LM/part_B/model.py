import torch.nn as nn

# Define the architecture of the first model (vanilla RNN)
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_RNN, self).__init__()

        # Define the network's layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) # Pytorch's RNN layer 
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)

        return output 

# Define the architecture of the second model (LSTM)
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, out_dropout, emb_dropout, use_dropout, pad_index, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.use_dropout = use_dropout

        # Define the network's layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if self.use_dropout:
            self.emb_dropout = nn.Dropout(emb_dropout) # First dropout layer
            self.out_dropout = nn.Dropout(out_dropout) # Second dropout layer

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True) # Pytorch's LSTM layer 
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.use_dropout: emb = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        if self.use_dropout: lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)

        return output 