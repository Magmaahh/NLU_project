import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import copy
import csv

from model import *

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Updates NT-AvSGD state and stores parameters for averaging when conditions are met
def avsgd_step(model, l_interval, n_interval, avsgd_state, dev_loader, criterion_eval):
    # At the end of each epoch (every l_interval steps), check if averaging should be triggered
    if avsgd_state["step"] % l_interval == 0 and avsgd_state["T"] is None:
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        model.train()
        # If enough logs were stored (t > n_interval) and performance has degraded compared to first n_interval results, trigger weight averaging
        if avsgd_state["t"] > n_interval:
            past_ppls = avsgd_state["logs"][:avsgd_state["t"] - n_interval]
            if past_ppls and ppl_dev > min(past_ppls):
                avsgd_state["T"] = avsgd_state["step"]
                print(f"Averaging triggered at step {avsgd_state['T']} with validation PPL {ppl_dev:.2f}")
        avsgd_state["logs"].append(ppl_dev)
        avsgd_state["t"] += 1

    # If averaging has been triggered, accumulate model parameters for averaging
    if avsgd_state["T"] is not None:
        with torch.no_grad():
            model_params = [param.detach().clone() for param in model.parameters()]
            if avsgd_state["avg_weights"] is None:
                avsgd_state["avg_weights"] = model_params
            else:
                for i in range(len(avsgd_state["avg_weights"])):
                    avsgd_state["avg_weights"][i] += model_params[i]
        avsgd_state["avg_count"] += 1

    avsgd_state["step"] += 1

# Trains the model for one epoch over the provided data
def train_loop(data, optimizer, criterion, model, use_avsgd, avsgd_state, dev_loader, criterion_eval, clip=5):
    loss_array = []
    number_of_tokens = []

    # NT-AvSGD hyperparameters
    l_interval = len(data)
    n_interval = 5

    model.train()
    for sample in data:
        optimizer.zero_grad()
        device = next(model.parameters()).device
        source = sample['source'].to(device)
        target = sample['target'].to(device)

        output = model(source)
        loss = criterion(output, target)

        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if use_avsgd:
            avsgd_step(model, l_interval, n_interval, avsgd_state, dev_loader, criterion_eval)

    return sum(loss_array) / sum(number_of_tokens)

# Evaluates the model over the provided data
def eval_loop(data, criterion, model):
    model.eval()  # Set model to evaluation mode
    loss_array = []
    number_of_tokens = []
    
    with torch.no_grad():  # Disable gradient computation
        for sample in data:
            output = model(sample['source']) # Forward pass
            loss = criterion(output, sample['target']) # Compute loss
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens)) # Compute perplexity: exp(mean loss per token)
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return

# Runs the whole training process for the model and provides a final evaluation on its performances
def train_model(model, train_loader, dev_loader, test_loader, criterion_train, criterion_eval, optimizer, params, use_avsgd):
    results = {
        "best_model": None,
        "losses_train": [],
        "losses_dev": [],
        "sampled_epochs": [],
        "ppls_dev": [],
        "best_ppl": math.inf,
        "final_ppl": None,
    }
    avsgd_state = {
        "step": 0,
        "T": None,
        "t": 0,
        "logs": [],
        "avg_weights": None,
        "avg_count": 0,
    }
    patience = params["patience_init"]
    pbar = tqdm(range(1, params["n_epochs"] + 1))

    # Full training loop
    print("\n==================== Training... ====================")
    for epoch in pbar:
        # Train on test data
        loss = train_loop(train_loader, optimizer, criterion_train, model, use_avsgd, avsgd_state, dev_loader, criterion_eval, clip=params["clip"])
        results["sampled_epochs"].append(epoch)
        results["losses_train"].append(np.asarray(loss).mean())

        # Evaluate on validation data
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        results["losses_dev"].append(np.asarray(loss_dev).mean())
        results["ppls_dev"].append(ppl_dev)

        pbar.set_description(f"Epoch {epoch} | Dev PPL: {ppl_dev:.2f} | Train Loss: {results['losses_train'][-1]:.2f}")

        # Patience-based early stopping logic
        if not use_avsgd:
            if ppl_dev < results["best_ppl"]:
                results["best_ppl"] = ppl_dev
                results["best_model"] = copy.deepcopy(model).to('cpu')
                patience = params["patience_init"]
            else:
                patience -= 1

            if patience <= 0:
                break

    # If averaging was triggered, average the stored weights to obtain the final values
    if use_avsgd and avsgd_state["avg_weights"] is not None:
        with torch.no_grad():
            for param, avg in zip(model.parameters(), avsgd_state["avg_weights"]):
                param.data = avg / avsgd_state["avg_count"]
        results["best_model"] = copy.deepcopy(model).to('cpu')
        print(f"Averaged over {avsgd_state['avg_count']} steps starting from step {avsgd_state['T']}.")
    else:
        results["best_model"] = copy.deepcopy(model).to('cpu')

    # Evaluate on test data
    results["best_model"].to(DEVICE)
    results["final_ppl"], _ = eval_loop(test_loader, criterion_eval, results["best_model"])
    print('Test PPL: ', results["final_ppl"])

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
def init_model(lang, vocab_len, params, configs):
    model = LM_LSTM(
        params["emb_size"], params["hid_size"], vocab_len,
        params["dropout"], configs["use_weight_tying"],
        configs["use_var_dropout"], pad_index=lang.word2id["<pad>"]
    ).to(DEVICE)
    
    return model

# Loads an existing model from the provided path
def load_model(model_path, lang, vocab_len, configs):
    print("\Loading the existing model...\n")
    saved_data = torch.load(model_path, map_location=DEVICE)
    model_state_dict = saved_data['model_state_dict']
    ref_params = saved_data['params']
    ref_model = init_model(lang, vocab_len, ref_params, configs)
    ref_model.load_state_dict(model_state_dict)

    return ref_model

# Allows the user to set the desired mode and model configuration
def select_config(configs):
    mode_input = input("Train or test mode? [train/test]: ").strip().lower()
    if mode_input not in {"train", "test"}:
        print("Invalid input. Defaulting to test mode.")
        mode_input = "test"
    configs["training"] = mode_input == "train"

    config_options = [
        "Basic LSTM",
        "LSTM + weight tying",
        "LSTM + weight tying + variational dropout",
        "LSTM + weight tying + variational dropout + NT-AvSGD"
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
    configs["use_weight_tying"] = choice in {"1", "2", "3"}
    configs["use_var_dropout"] = choice in {"2", "3"}
    configs["use_avsgd"] = choice == "3"

    # Print summary
    print("\n==================== Selected Configuration ====================")
    print(f"Mode: {'Training' if configs['training'] else 'Testing'}")
    print("Model: LSTM")
    print(f"Weight tying: {'Enabled' if configs['use_weight_tying'] else 'Disabled'}")
    print(f"Variational dropout: {'Enabled' if configs['use_var_dropout'] else 'Disabled'}")
    print(f"NT-AvSGD: {'Enabled' if configs['use_avsgd'] else 'Disabled'}")
    print("===============================================================\n")

# Returns a string summarizing the selected model configuration
def get_config(configs):
    parts = []
    if configs["use_weight_tying"]:
        parts.append("weight tying")
    if configs["use_var_dropout"]:
        parts.append("variational dropout")
    if configs["use_avsgd"]:
        parts.append("NT-AvSGD")
        
    return "LSTM" + (" + " + " + ".join(parts) if parts else "")

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

# Logs and plots training results
def log_and_plot_results(configs, params, results, log_path, plot_path):
    # Create a unique experiment ID based on config count
    config = get_config(configs)
    config_count = 0
    if os.path.exists(log_path):
        with open(log_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model_config"] == config:
                    config_count += 1

    # Plot and save training results
    os.makedirs(plot_path, exist_ok=True)
    config_id = f"{config}_v{config_count + 1}"
    plot_filename = f"{config_id}_loss_plot.png"
    plot_filepath = os.path.join(plot_path, plot_filename)
    
    plt.figure()
    plt.plot(results["sampled_epochs"], results["losses_train"], label="Training Loss", marker="o")
    plt.plot(results["sampled_epochs"], results["losses_dev"], label="Dev Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Dev Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.close()

    # Log and save training results
    log_fields = [
        "experiment_id", "model_config", "lr", "training_batch_size", "hid_size", 
        "emb_size", "dropout", "dev_PPL", "test_PPL", "notes"
    ]
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writerow({
            "experiment_id": config_id,
            "model_config": config,
            "lr": params["lr"],
            "training_batch_size": params["tr_batch_size"],
            "hid_size": params["hid_size"],
            "emb_size": params["emb_size"],
            "dropout": params["dropout"],
            "dev_PPL": results["best_ppl"],
            "test_PPL": results["final_ppl"],
            "notes": ""
        })