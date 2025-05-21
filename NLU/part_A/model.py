import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Define the architecture of the model
class ModelIAS(nn.Module):
    def __init__(self, hid_size, emb_size, out_slot, out_int, vocab_len, dropout, bidirectional, apply_dropout, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        self.apply_dropout = apply_dropout
        self.bidirectional = bidirectional
        
        # Define the network's layers
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=self.bidirectional, batch_first=True) 

        if self.apply_dropout:
            self.dropout = nn.Dropout(dropout)   
        
        if self.bidirectional: # Double the hidden layer's size to account for bidirectionality
            self.slot_out = nn.Linear(hid_size * 2, out_slot) 
            self.intent_out = nn.Linear(hid_size * 2, out_int)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)
            self.intent_out = nn.Linear(hid_size, out_int)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)

        if self.apply_dropout:
            utt_emb = self.dropout(utt_emb)
        
        # Pack sequences, process them and unpack them in the end
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input) 
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1) # Concatenate the last forward and backward hidden states
        else:
            last_hidden = last_hidden[-1,:,:]

        if self.apply_dropout:
            utt_encoded = self.dropout(utt_encoded)
            last_hidden = self.dropout(last_hidden)
        
        slots = self.slot_out(utt_encoded) # Slot filling
        slots = slots.permute(0,2,1)

        intent = self.intent_out(last_hidden) # Intent classification

        return slots, intent