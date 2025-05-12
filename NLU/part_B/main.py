import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy
import csv
from collections import Counter
from transformers import BertConfig

from model import *
from utils import *
from functions import *

# Changable constants to try all configurations
LARGE_MODEL = False

# Training hyperparameters
dropout = 0.1 # default 0.1
lr = 0.00005 # default 0.00005
batch_size_train = 128 # default 128
clip = 5
patience_init = 3
n_epochs = 40
runs = 1
bert_model = "bert-large-uncased" if LARGE_MODEL else "bert-base-uncased"

# Open log
log_path = "experiment_log.csv"
log_fields = [
    "model_id", "lr", "training_batch_size", "slot_f1", "intent_acc", "notes"
]
os.makedirs('./bin', exist_ok=True)
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

if __name__ == "__main__":
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))

    intents = [x['intent'] for x in tmp_train_raw] 
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: 
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=0.10, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]
                                                  
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(intents, slots, cutoff=0)
    
    # create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang, bert_model)
    dev_dataset = IntentsAndSlots(dev_raw, lang, bert_model)
    test_dataset = IntentsAndSlots(test_raw, lang, bert_model)

    # create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size_train, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    config = BertConfig.from_pretrained(bert_model)

    slot_f1s, intent_acc = [], []
    
    for x in tqdm(range(0, runs)):
        model = BERTmodel(config, out_slot, out_int, dropout).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        criterion_intents = nn.CrossEntropyLoss()
        
        patience = patience_init
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0

        for x in tqdm(range(1,n_epochs)):
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                if x % 5 == 0: 
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, bert_model, testing=False)
                    losses_dev.append(np.asarray(loss_dev).mean())
                    
                    f1 = results_dev['total']['f']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = 3
                    else:
                        patience -= 1
                    if patience <= 0:
                        break 
            
        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang, bert_model, testing=True)   
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f']) 
            
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    slot_f1 = round(slot_f1s.mean(),3)
    intent_accuracy = round(intent_acc.mean(), 3)

    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    if os.path.exists(log_path):
        with open(log_path, mode="r") as f:
            reader = csv.reader(f)
            next(reader)
            row_count = sum(1 for _ in reader) + 1
    else:
        row_count = 1

    model_id = f"{row_count:03d}"

    # Log model's results  
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "model_id": model_id,
            "lr": lr,
            "training_batch_size": batch_size_train,
            "slot_f1": slot_f1,
            "intent_acc": intent_accuracy,
            "notes": ""
        })

    MODEL_PATH = os.path.join("bin", f"{model_id}.pt")
    saving_object = {"epoch": x, 
                    "model": best_model.state_dict(), 
                    "optimizer": optimizer.state_dict(), 
                    "slot2id": lang.slot2id, 
                    "intent2id": lang.intent2id}
    torch.save(saving_object, MODEL_PATH)
    
    plot_data(model_id, sampled_epochs, losses_train, losses_dev)