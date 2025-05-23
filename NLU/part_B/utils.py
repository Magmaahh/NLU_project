import torch
import torch.utils.data as data
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import Counter
from transformers import AutoTokenizer

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Pad token for vocabulary preparation
PAD_TOKEN = 0

# Computes and stores the vocabulary
class Lang():
    def __init__(self, intents, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

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
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        for el in dataset:
            self.utterances.append(el['utterance'])
            self.slots.append(el['slots'])
            self.intents.append(el['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.LongTensor(self.utt_ids[idx])
        slots = torch.LongTensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    # Maps sequences to numbers while dealing with BERT subtokenization issue
    def mapping_seq(self, utterance_list, slot_list, mapper):
        utt_ids = []
        slot_ids = []

        for utt, slots in zip(utterance_list, slot_list):
            slot_labels = slots.split()
            encoding = self.tokenizer(utt.split(), is_split_into_words=True, return_tensors=None, truncation=True)

            word_ids = encoding.word_ids()
            input_ids = encoding['input_ids']
            
            aligned_slot_ids = []
            previous_word_id = None
            label_id = 0

            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    aligned_slot_ids.append(PAD_TOKEN)  # Special tokens like [CLS], [SEP]
                elif word_id != previous_word_id:
                    # First subtoken of a new word
                    aligned_slot_ids.append(mapper.get(slot_labels[label_id], PAD_TOKEN))
                    previous_word_id = word_id
                    label_id += 1
                else:
                    # Subsequent subtokens of the same word
                    aligned_slot_ids.append(PAD_TOKEN)

            utt_ids.append(input_ids)
            slot_ids.append(aligned_slot_ids)

        return utt_ids, slot_ids

# Pads sequences and batches them
def collate_fn(data):
    def merge(sequences, pad_token):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    utts, _ = merge(new_item['utterance'], PAD_TOKEN)
    attention_mask = torch.LongTensor([[1 if id != PAD_TOKEN else 0 for id in seq] for seq in utts])
    y_slots, _ = merge(new_item["slots"], PAD_TOKEN)
    intent = torch.LongTensor(new_item["intent"])
    
    utts = utts.to(DEVICE) 
    intent = intent.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    
    new_item["utterances"] = utts
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["attention_mask"] = attention_mask
    
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
def prepare_data(train_path, test_path, model_path, params, training):
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)
    train_raw, dev_raw = create_raws(tmp_train_raw)

    if training:                                                  
        corpus = train_raw + dev_raw + test_raw
        slots = set(sum([line['slots'].split() for line in corpus],[]))
        intents = set([line['intent'] for line in corpus])
        lang = Lang(intents, slots, cutoff=0)
    else:
        if os.path.exists(model_path):
            saved_data = torch.load(model_path, map_location=DEVICE)
            lang = Lang([], [], cutoff=0)
            lang.slot2id = saved_data['slot2id']
            lang.intent2id = saved_data['intent2id']
            lang.id2slot = {v: k for k, v in lang.slot2id.items()}
            lang.id2intent = {v: k for k, v in lang.intent2id.items()}
        else:
            print(f"Error: No model for the selected config is saved. Exiting.")
            exit(1)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    # Create datasets and loaders
    train_dataset, dev_dataset, test_dataset = create_datasets(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = create_dataloaders(train_dataset, dev_dataset, test_dataset, params["tr_batch_size"])

    return train_loader, dev_loader, test_loader, lang, out_slot, out_int