import torch
import torch.nn as nn
import os
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

def eval_loop(data, criterion_slots, criterion_intents, model, lang, bert_model):
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