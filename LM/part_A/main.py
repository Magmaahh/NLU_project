import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import math
import numpy as np
from tqdm import tqdm
import copy
import csv

from model import *
from utils import *
from functions import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
out_dropout = 0.1
emb_dropout = 0.1
lr = 0.1
batch_size_train = 64 # default 64
hid_size = 200 # default 200
emb_size = 300 # defaul 300
clip = 5
n_epochs = 100
patience_init = 3

# Constants to try out different configurations
USE_LSTM = True
USE_DROPOUT = False
USE_ADAMW = False

# Reference PPL to analyze model's performances
REFERENCE_PPL = 250

# Open log
log_path = "experiment_log.csv"
log_fields = [
    "model_id", "optimizer", "lr", "training_batch_size", "hid_size", 
    "emb_size", "slot_f1", "intent_acc", "notes"
]
os.makedirs('./bin', exist_ok=True)
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

if __name__ == "__main__":
    # Data instantiation
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # Model instantiation
    if USE_LSTM:
        model = LM_LSTM(emb_size, hid_size, vocab_len, emb_dropout, out_dropout, USE_DROPOUT, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)
    else:
        model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)
        
    if USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Model training
    patience = patience_init
    losses_train, losses_dev, sampled_epochs, ppls_dev = [], [], [], []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs + 1))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())

        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(np.asarray(loss_dev).mean())
        ppls_dev.append(ppl_dev)

        pbar.set_description(f"Epoch {epoch} | Dev PPL: {ppl_dev:.2f} | Train Loss: {losses_train[-1]:.2f}")

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = patience_init
        else:
            patience -= 1

        if patience <= 0:
            break

    # Model evaluation
    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test PPL: ', final_ppl)

    if os.path.exists(log_path):
        with open(log_path, mode="r") as f:
            reader = csv.reader(f)
            next(reader)
            row_count = sum(1 for _ in reader) + 1
    else:
        row_count = 1

    model_id = f"{row_count:03d}"

    # Save the model if final perplexity is below the reference value
    if final_ppl <= REFERENCE_PPL:
        torch.save(best_model.state_dict(), os.path.join("bin", f"{model_id}.pt"))

    # Log model's results  
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "model_id": model_id,
            "optimizer": "SGD" if not USE_ADAMW else "AdamW",
            "lr": lr,
            "training_batch_size": batch_size_train,
            "hid_size": hid_size,
            "emb_size": emb_size,
            "dev_PPL": best_ppl,
            "test_PPL": final_ppl,
            "notes": ""
        })

    plot_data(model_id, sampled_epochs, losses_train, losses_dev, ppls_dev)
  
    '''TO DO:
    add TRAINING and TESTING modes (EXECUTABLES)
       TRAINING -> Select config, select hyperparams values, train, save model in bin if the best of the config
       TESTING -> Select set of weight from bin folder, evaluate it
    '''