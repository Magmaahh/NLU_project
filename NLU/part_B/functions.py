import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from conll import evaluate
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from utils import IGNORE_INDEX

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []

    for sample in data:
        optimizer.zero_grad() 
        slots, intent = model(sample['utterances'], sample['attention_mask'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 
        loss_array.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang, bert_model, testing):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    with torch.no_grad():
        for batch_id, sample in enumerate(data):
            slots, intent = model(sample['utterances'], sample['attention_mask'])
            loss_intent = criterion_intents(intent, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
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
                    if gt_id != IGNORE_INDEX:
                        ref_slots_temp.append(lang.id2slot[gt_id])
                        hyp_slots_temp.append(lang.id2slot[seq[i].item()])
                        utterance_temp.append(utterance[i]) 

                if testing:
                    diff_found = False
                    for ref, hyp in zip(ref_slots_temp, hyp_slots_temp):
                        if ref != hyp:
                            diff_found = True
                            print(f"Difference found - Ref: {ref}, Hyp: {hyp}")
                    
                    if diff_found:
                        print(f"Utterance: {utterance_temp}")
                        print(f"Refs: {ref_slots_temp}")
                        print(f"Hyps: {hyp_slots_temp}")

                ref_slots.append(list(zip(utterance_temp, ref_slots_temp)))
                hyp_slots.append(list(zip(utterance_temp, hyp_slots_temp)))
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning during evaluation:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print("Mismatched slot tags:", hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

    return results, report_intent, loss_array

def init_weights(mat):
    for m in mat.modules():
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

def plot_data(model_id, epochs, losses_train, losses_dev):
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