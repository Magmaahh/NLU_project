import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import copy
from conll import evaluate
from sklearn.metrics import classification_report
import csv

from model import *
from utils import PAD_TOKEN

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Trains the model for one epoch over the provided data
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train() # Set model to training mode
    loss_array = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len']) # Forward pass
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # Compute loss; since it's joint training, it is the sum of the individual losses
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # clip the gradient to avoid exploding gradients
        optimizer.step() # Update the weights

    return loss_array

# Evaluates the model over the provided data
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval() # Set the model to evaluation mode
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): # Disable gradient computation
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len']) # Forward pass
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot # Compute loss
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterances'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s)) 
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

# Runs the whole training process for the model for multiple times (= runs) and provides a final evaluation on its average performances
def train_model(train_loader, dev_loader, test_loader, lang, out_int, out_slot, vocab_len, criterion_slots, criterion_intents, params, configs):
    results = {
        "best_model": None,
        "slot_f1": 0,
        "int_acc": 0
    }
    slot_f1s, intent_acc, best_models = [], [], []

    for run in tqdm(range(0, params["runs"])):
        model = init_model(out_slot, out_int, vocab_len, params, configs)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        patience = params["patience_init"]
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None

        for epoch in tqdm(range(1,params["n_epochs"])):
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=params["clip"])
                if epoch % 5 == 0: 
                    sampled_epochs.append(epoch)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
                    losses_dev.append(np.asarray(loss_dev).mean())
                    
                    f1 = results_dev['total']['f']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = copy.deepcopy(model).to('cpu')
                        patience = params["patience_init"]
                    else:
                        patience -= 1
                    if patience <= 0:
                        break 
        
        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)   
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        best_models.append((best_model, best_f1))
            
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    results["slot_f1"] = round(slot_f1s.mean(),3)
    results["int_acc"] = round(intent_acc.mean(), 3)
    best_model, _ = max(best_models, key=lambda x: x[1])
    results["best_model"] = copy.deepcopy(best_model).to('cpu')

    print('Slot F1', results['slot_f1'], '+-', round(slot_f1s.std(),3))
    print('Intent Acc', results['int_acc'], '+-', round(slot_f1s.std(), 3))

    return results


# Initializes the weights of the model layers
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

# Initializes the model with the provided settings
def init_model(out_slot, out_int, vocab_len, params, configs):
    model = ModelIAS(
        params["hid_size"], params["emb_size"], out_slot, out_int,
        vocab_len, params["dropout"], configs["use_bidir"], configs["use_dropout"], pad_index=PAD_TOKEN
    ).to(DEVICE)
    
    return model

# Loads an existing model from the provided path
def load_model_data(model_path, out_int, out_slot, vocab_len, configs):
    print("\nLoading the existing model...\n")
    saved_data = torch.load(model_path, map_location=DEVICE)
    ref_model = init_model(out_slot, out_int, vocab_len, saved_data['params'], configs)
    ref_model.load_state_dict(saved_data['model_state_dict'])
    ref_model.to(DEVICE)

    return ref_model

# Allows the user to set the desired mode and model configuration
def select_config(configs):
    mode_input = input("Train or test mode? [train/test]: ").strip().lower()
    if mode_input not in {"train", "test"}:
        print("Invalid input. Defaulting to test mode.")
        mode_input = "test"
    configs["training"] = mode_input == "train"

    config_options = [
        "Basic model",
        "Bidirectional",
        "Bidirectional + dropout",
    ]
    print("Choose model configuration:")
    start_idx = 0 if configs["training"] else 1
    for idx in range(start_idx, len(config_options)):
        print(f"{idx}. {config_options[idx]}")
    valid_choices = {str(i) for i in range(start_idx, len(config_options))}

    choice = None
    while choice not in valid_choices:
        choice = input(f"Enter your choice between {sorted(valid_choices)}: ").strip()
        if choice not in valid_choices:
            print(f"Invalid choice. Please select one between {sorted(valid_choices)}.")
    configs["use_bidir"] = choice in {"1", "2"}
    configs["use_dropout"] = choice in {"2"}

    # Print summary
    print("\n==================== Selected Configuration ====================")
    print(f"Mode: {'Training' if configs['training'] else 'Testing'}")
    print(f"Bidirectionality: {'Enabled' if configs['use_bidir'] else 'Disabled'}")
    print(f"Dropout: {'Enabled' if configs['use_dropout'] else 'Disabled'}")
    print("===============================================================\n")

# Returns a string summarizing the selected model configuration
def get_config(configs):
    parts = []
    if configs["use_bidir"]:
        parts.append("Bidirectional")
    if configs["use_dropout"]:
        parts.append("Dropout")
        
    return " + ".join(parts) if parts else "Basic model"

# Casts a string value to a specified type with error handling
def cast_value(value, to_type):
    try:
        return to_type(value)
    except ValueError:
        print(f"Invalid type: expected {to_type.__name__}. Please try again.")
        return None

# Prints the current parameters' values
def print_params(params):
    print("\nCurrent parameters values:")
    for k, v in params.items():
        print(f"  {k}: {v}")

# Allows the user to modify training hyperparameters via terminal input
def select_params(params):
    print_params(params)
    choice_input = input("\nWould you like to change any of the parameters above? [y/n]: ").strip().lower()
    if choice_input not in {"y", "n"}:
        print("Invalid input. Defaulting to no.")
        choice_input = "n"
    changing = choice_input == "y"

    while changing:
        key = input("Enter the parameter name to change (e.g., lr, hid_size): ").strip()
        if key not in params:
            print(f"'{key}' is not a valid parameter. Please insert a valid one.")
        else:
            new_val = input(f"Enter new value for '{key}' (current: {params[key]}): ").strip()
            casted_val = cast_value(new_val, type(params[key]))
            if casted_val is not None:
                params[key] = casted_val
                print(f"Updated '{key}' to {casted_val}\n")
                changing = False

        if not changing:
            print_params(params)
            more = input("Change another parameter? [y/n]: ").strip().lower()
            changing = more == "y"

# Logs the training configuration and results into a CSV file
def log_results(configs, params, results, log_path):
    log_fields = [
        "model_config", "lr", "training_batch_size", "hid_size", 
        "emb_size", "dropout", "slot_f1", "int_accuracy", "notes"
    ]
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "model_config": get_config(configs),
            "lr": params["lr"],
            "training_batch_size": params["tr_batch_size"],
            "hid_size": params["hid_size"],
            "emb_size": params["emb_size"],
            "dropout": params["dropout"],
            "slot_f1": results["slot_f1"],
            "int_accuracy": results["int_acc"],
            "notes": ""
        })