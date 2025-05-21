import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Pad token for vocabulary preparation
PAD_TOKEN = 0

# Computes and stores the vocabulary
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    # Create word2id dictionary
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    # Create lab2id dictionary
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

# Provides ID versions of the datasets
class IntentsAndSlots (data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Maps sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

# Pads sequences and batches them
def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

# Loads data from the provided path
def load_data(path):
    dataset = []

    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

# Creates validation raw data and final test raw data
def create_raws(tmp_train_raw):
    labels = []
    inputs = []
    mini_train = []

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    portion = 0.10

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw

# Creates dataset objects from raw data
def create_datasets(train_raw, dev_raw, test_raw, lang):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset

# Creates DataLoader objects with padding and batching
def create_dataloaders(train_dataset, dev_dataset, test_dataset, train_batch_size):
    train_loader = DataLoader(train_dataset, train_batch_size, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
                             
    return train_loader, dev_loader, test_loader

# Prepares raw data, vocabulary, datasets and dataloaders
def prepare_data(train_path, test_path, model_path, params, configs):
    # Extract raw data and create raw validation set
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)
    train_raw, dev_raw = create_raws(tmp_train_raw)

    # Create the vocabulary
    words = sum([x['utterance'].split() for x in train_raw], [])                                                      
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    if not configs['training']:
        saved_data = torch.load(model_path, map_location=DEVICE)
        lang.word2id = saved_data['w2id']
        lang.slot2id = saved_data['slot2id']
        lang.intent2id = saved_data['intent2id']

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    # Create the datasets
    train_dataset, dev_dataset, test_dataset = create_datasets(train_raw, dev_raw, test_raw, lang)

    # create the dataloaders
    train_loader, dev_loader, test_loader = create_dataloaders(train_dataset, dev_dataset, test_dataset, params["tr_batch_size"])

    return train_loader, dev_loader, test_loader, lang, out_slot, out_int, vocab_len