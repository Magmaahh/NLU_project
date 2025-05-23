import numpy as np
import torch
import torch.nn as nn
import os
import csv
from tqdm import tqdm
import copy
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertConfig

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
        slots, intent = model(sample['utterances'], sample['attention_mask']) # Forward pass
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    with torch.no_grad(): # Disable gradient computation
        for batch_id, sample in enumerate(data):
            slots, intent = model(sample['utterances'], sample['attention_mask']) # Forward pass
            loss_intent = criterion_intents(intent, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot # Compute loss
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)

            for id_seq, seq in enumerate(output_slots):
                utt_ids = sample['utterances'][id_seq].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()

                utterance = tokenizer.convert_ids_to_tokens(utt_ids)
                
                ref_slots_temp = []
                hyp_slots_temp = []
                utterance_temp = []
                for i, gt_id in enumerate(gt_ids):
                    # Skip special tokens's cells
                    if gt_id != PAD_TOKEN:
                        ref_slots_temp.append(lang.id2slot[gt_id])
                        hyp_slots_temp.append(lang.id2slot[seq[i].item()])
                        utterance_temp.append(utterance[i]) 

                ref_slots.append(list(zip(utterance_temp, ref_slots_temp)))
                hyp_slots.append(list(zip(utterance_temp, hyp_slots_temp)))
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning during evaluation:", ex)
        ref_s = set(tag for seq in ref_slots for (_, tag) in seq)
        hyp_s = set(tag for seq in hyp_slots for (_, tag) in seq)
        print("Mismatched slot tags:", hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

    return results, report_intent, loss_array

# Runs the whole training process for the model for multiple times (= runs) and provides a final evaluation on its average performances
def train_model(train_loader, dev_loader, test_loader, lang, out_int, out_slot, criterion_slots, criterion_intents, params):
    results = {
        "best_model": None,
        "slot_f1": 0,
        "int_acc": 0,
        "losses_dev": [],
        "losses_train": [],
        "sampled_epochs": [],
    }
    slot_f1s, intent_acc, best_models = [], [], []
    
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    
    for run in tqdm(range(0, params["runs"])):
        model = init_model(bert_config, out_slot, out_int, params)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        
        patience = params["patience_init"]
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None

        for x in tqdm(range(1,params["n_epochs"])):
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
                if x % 5 == 0: 
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, _, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
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
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)   
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        best_models.append((best_model, best_f1))
        results["losses_dev"].append(losses_dev)
        results["losses_train"].append(losses_train)
        results["sampled_epochs"].append(sampled_epochs)

        print('Slot F1', results_test['total']['f'])
        print('Intent Acc', intent_test['accuracy'])

    # Compute and store mean for slot_f1s and intent accuracy
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
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)    

# Initializes the model with the provided settings
def init_model(bert_config, out_slot, out_int, params):
    model = BERTmodel(bert_config, out_slot, out_int, params["dropout"]).to(DEVICE)
    
    return model

# Loads an existing model from the provided path
def load_model_data(model_path, out_int, out_slot):
    bert_config = BertConfig.from_pretrained("bert-base-uncased")

    print("\nLoading the existing model...\n")
    saved_data = torch.load(model_path, map_location=DEVICE)
    ref_model = init_model(bert_config, out_slot, out_int, saved_data["params"])
    ref_model.load_state_dict(saved_data['model_state_dict'])
    ref_model.to(DEVICE)

    return ref_model

# Allows the user to set the desired mode configuration
def select_config():
    mode_input = input("Train or test mode? [train/test]: ").strip().lower()
    if mode_input not in {"train", "test"}:
        print("Invalid input. Defaulting to test mode.")
        mode_input = "test"
    training = mode_input == "train"

    # Print summary
    print("\n==================== Selected Configuration ====================")
    print(f"Mode: {'Training' if training else 'Testing'}")
    print("===============================================================\n")

    return training

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
        key = input("Enter the parameter name to change (e.g., lr, dropout): ").strip()
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

# Creates a unique experiment ID based on config count in the log file
def get_experiment_id(log_path):
    config = "bert-base-uncased"
    config_count = 0
    if os.path.exists(log_path):
        with open(log_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model_config"] == config:
                    config_count += 1

    config_id = f"{config}_v{config_count + 1}"

    return config_id

# Plots training results
def plot_results(results, log_path, plot_path):
    # Get the experiment id and make a subfolder for plots
    config_id = get_experiment_id(log_path)
    folder_name = os.path.join(plot_path, config_id)
    os.makedirs(folder_name, exist_ok=True)

    # Plot one graph per run
    for run_idx, (epochs, train_losses, dev_losses) in enumerate(
        zip(results["sampled_epochs"], results["losses_train"], results["losses_dev"])
    ):
        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(epochs, dev_losses, label="Dev Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Run {run_idx + 1}: Training and Dev Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"{config_id}_run_{run_idx + 1}.png"
        filepath = os.path.join(folder_name, filename)
        plt.savefig(filepath)
        plt.close()

# Logs training results
def log_results(params, results, log_path):
    # Get the experiment id
    config_id = get_experiment_id(log_path)

    # Log and save training results
    log_fields = [
        "experiment_id", "model_type", "lr", "training_batch_size", "dropout", "slot_f1", "int_accuracy", "notes"
    ]
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "experiment_id": config_id,
            "model_type": "bert-base-uncased",
            "lr": params["lr"],
            "training_batch_size": params["tr_batch_size"],
            "dropout": params["dropout"],
            "slot_f1": results["slot_f1"],
            "int_accuracy": results["int_acc"],
            "notes": ""
        })