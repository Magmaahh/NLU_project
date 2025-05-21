import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Computes and stores the vocabulary
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    # Create word2id dictionary
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1

        return output

# Prepares input-target token sequences
class PennTreeBank (data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        
        return sample
    
    # Maps sequences of tokens to corresponding IDs using Lang class
    def mapping_seq(self, data, lang): 
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)

        return res

# Pads sequences and batches them
def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()

        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)

    return new_item

# Reads lines from the provided file and appends <eos> to each line
def read_file(path, eos_token="<eos>"):
    output=[]
    with open(path, "r") as file:
        for line in file.readlines():
            output.append(line.strip() + " " + eos_token)

    return output

# Loads raw data from the provided files
def load_data(train_data_path, dev_data_path, test_data_path):
    train_raw = read_file(train_data_path)
    dev_raw = read_file(dev_data_path)
    test_raw = read_file(test_data_path)

    return train_raw, dev_raw, test_raw

# Creates dataset objects from raw data
def create_datasets(train_raw, dev_raw, test_raw, lang):
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset

# Creates DataLoader objects with padding and batching
def create_dataloaders(train_dataset, dev_dataset, test_dataset, lang, train_batch_size):
    pad_token = lang.word2id["<pad>"]
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              collate_fn=partial(collate_fn, pad_token=pad_token), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128,
                            collate_fn=partial(collate_fn, pad_token=pad_token))
    test_loader = DataLoader(test_dataset, batch_size=128,
                             collate_fn=partial(collate_fn, pad_token=pad_token))
                             
    return train_loader, dev_loader, test_loader

# Prepares raw data, vocabulary, datasets and dataloaders
def prepare_data(train_path, dev_path, test_path, params):
    train_raw, dev_raw, test_raw = load_data(train_path, dev_path, test_path)
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    train_dataset, dev_dataset, test_dataset = create_datasets(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = create_dataloaders(train_dataset, dev_dataset, test_dataset, lang, params["tr_batch_size"])

    return train_loader, dev_loader, test_loader, lang, vocab_len