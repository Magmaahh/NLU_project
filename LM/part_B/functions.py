import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def train_loop(data, optimizer, criterion, model, clip=5, use_avsgd=False, dev_loader=None, criterion_eval=None, avsgd_state=None):
    model.train()
    loss_array = []
    number_of_tokens = []

    # NT-AvSGD hyperparameters
    l_interval = len(data) # default value = number of iterations in an epoch
    n_interval = 5 # default 5

    if use_avsgd:
        step = avsgd_state["step"]
        t = avsgd_state["t"]
        T = avsgd_state["T"]
        avg_weights = avsgd_state["avg_weights"]
        avg_count = avsgd_state["avg_count"]
        logs = avsgd_state["logs"]
    else:
        step = 0
        t = 0
        T = None
        avg_weights = None
        avg_count = 0
        logs = []

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
            if step % l_interval == 0 and T is None:
                ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
                model.train()
                if t > n_interval and ppl_dev > min(logs[:t-n_interval]):
                    T = step
                    print(f"Averaging triggered at step {T} with validation PPL {ppl_dev:.2f}")
                logs.append(ppl_dev)
                t += 1

            if T is not None:
                with torch.no_grad():
                    model_params = [param.detach().clone() for param in model.parameters()]
                    if avg_weights is None:
                        avg_weights = model_params
                    else:
                        for i in range(len(avg_weights)):
                            avg_weights[i] += model_params[i]
                avg_count += 1
        step = step + 1

    if use_avsgd:
        avsgd_state["step"] = step
        avsgd_state["t"] = t
        avsgd_state["T"] = T
        avsgd_state["avg_weights"] = avg_weights
        avsgd_state["avg_count"] = avg_count
        avsgd_state["logs"] = logs

    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:
            device = next(model.parameters()).device
            source = sample['source'].to(device)
            target = sample['target'].to(device)
            output = model(source)
            loss = eval_criterion(output, target)

            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array)/sum(number_of_tokens))
    loss_to_return = sum(loss_array)/sum(number_of_tokens)
    return ppl, loss_to_return 

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

def set_adaptive_ylim(y_data, margin=0.1, nbins=5):
    y_min = min(y_data)
    y_max = max(y_data)
    plt.ylim(y_min * (1 - margin), y_max * (1 + margin))
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))

def plot_data(model_id, epochs, losses_train, losses_dev, ppls_dev):
    os.makedirs("plots", exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("muted")

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses_train, label="Train Loss", marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.plot(epochs, losses_dev, label="Dev Loss", marker='s', linestyle='--', linewidth=2, markersize=6)
    set_adaptive_ylim(losses_train + losses_dev)
    plt.title(f"{model_id} - Training Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/{model_id}_loss.png", dpi=300)
    plt.close()

    # Perplexity plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, ppls_dev, label="Dev PPL", marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.axhline(y=250, linewidth=1.5, color='gray', linestyle='--', label='Reference PPL')
    set_adaptive_ylim(ppls_dev)
    plt.title(f"{model_id} - Validation Perplexity Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/{model_id}_perplexity.png", dpi=300)
    plt.close() 