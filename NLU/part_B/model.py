import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

# Define the architecture of the model
class BERTmodel(BertPreTrainedModel):
    def __init__(self, config, out_slot, out_int, dropout):
        super(BERTmodel, self).__init__(config)

        # Define the network's layers
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.slots_out = nn.Linear(config.hidden_size, out_slot)
        self.int_out = nn.Linear(config.hidden_size, out_int)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_out = output.last_hidden_state
        #cls_out = output.last_hidden_state[:, 0, :]
        cls_out = output.pooler_output

        # Apply dropout
        sequence_out = self.dropout(sequence_out)
        cls_out = self.dropout(cls_out)   
           
        slots = self.slots_out(sequence_out)
        slots = slots.permute(0, 2, 1)
        
        intent = self.int_out(cls_out)

        return slots, intent