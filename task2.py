# IMPORTS
import pandas as pd
import numpy as np
import re
import os
import random
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import transformers
import matplotlib.pyplot as plt

# CONFIG
SAVE_PATH = "./"
LOAD_LM = True

# SEED
seed_val = 1234
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# DEVICE
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using ", device)

# DATA
def get_sentence_pair(sent_orig, edit_word):
    sent_o = re.sub("[</>]", "", sent_orig)
    sent_e = (sent_orig.split("<"))[0] + edit_word + (sent_orig.split(">"))[1]

    return sent_e, sent_o


class two_sentence_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.df = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        self.cls = [self.tokenizer.cls_token_id]
        self.sep = [self.tokenizer.sep_token_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        grade = torch.tensor(self.df["meanGrade"][idx])

        sent_e, sent_o = get_sentence_pair(
            self.df["original"][idx], self.df["edit"][idx]
        )

        sent_e_tokens = (
            self.cls
            + self.tokenizer.encode(sent_e, add_special_tokens=False)
            + self.sep
        )
        sent_o_tokens = (
            self.tokenizer.encode(sent_o, add_special_tokens=False) + self.sep
        )

        token_type_ids = (
            torch.tensor(
                [0] * len(sent_e_tokens) + [1] * (self.max_len - len(sent_e_tokens))
            )
        ).long()
        attention_mask = torch.tensor(
            [1] * (len(sent_o_tokens) + len(sent_e_tokens))
            + [0] * (self.max_len - len(sent_o_tokens) - len(sent_e_tokens))
        )
        attention_mask = attention_mask.float()

        input_ids = torch.tensor(sent_o_tokens + sent_e_tokens)

        if len(input_ids) < self.max_len:
            input_ids = torch.cat(
                (
                    input_ids,
                    (torch.ones(self.max_len - len(input_ids)) * self.pad).long(),
                )
            )
            token_type_ids = torch.cat(
                (
                    token_type_ids,
                    (torch.ones(self.max_len - len(token_type_ids)) * self.pad).long(),
                )
            )
        elif len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]

        return input_ids, token_type_ids, attention_mask, grade, loss_weight


train_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/train.csv")
valid_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/dev.csv")
test_df = pd.read_csv("semeval-2020-task-7-data-full/task-2/test.csv")

x = train_df.append(valid_df, ignore_index=True)
total_data_df = x.append(test_df, ignore_index=True)

model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

total_df_1 = total_data_df[["id", "original1", "edit1", "meanGrade1"]]
total_df_2 = total_data_df[["id", "original2", "edit2", "meanGrade2"]]
total_df_1.rename(
    columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"},
    inplace=True,
)
total_df_2.rename(
    columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"},
    inplace=True,
)

total_dset_1 = two_sentence_dataset(total_df_1, tokenizer, max_len=256)
total_dset_2 = two_sentence_dataset(total_df_2, tokenizer, max_len=256)

total_loader_1 = DataLoader(total_dset_1, batch_size=32, shuffle=False, num_workers=0)
total_loader_2 = DataLoader(total_dset_2, batch_size=32, shuffle=False, num_workers=0)

# MODEL
model = transformers.BertForSequenceClassification.from_pretrained(
    model_name, num_labels=1
)
model = model.to(device)

if LOAD_LM:
    fname = "without_LM" + model_name + "_task1"
else:
    fname = model_name + "_task1"

# PREDICTIONS ON TEST
preds_1 = []
model.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(total_loader_1):
        input_ids, token_type_ids, attention_mask, grade = data
        input_ids, token_type_ids, attention_mask, grade = (
            input_ids.to(device),
            token_type_ids.to(device),
            attention_mask.to(device),
            grade.to(device),
        )
        outputs = model(
            input_ids=input_ids,
            labels=grade,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _, logits = outputs[:2]

        for logit in logits.reshape(-1):
            preds_1.append(logit.item())

preds_2 = []
model.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(total_loader_2):
        input_ids, token_type_ids, attention_mask, grade = data
        input_ids, token_type_ids, attention_mask, grade = (
            input_ids.to(device),
            token_type_ids.to(device),
            attention_mask.to(device),
            grade.to(device),
        )
        outputs = model(
            input_ids=input_ids,
            labels=grade,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _, logits = outputs[:2]

        for logit in logits.reshape(-1):
            preds_2.append(logit.item())

final_preds = []
for i in range(len(preds_1)):
    if preds_1[i] > preds_2[i]:
        final_preds.append(1)
    elif preds_2[i] > preds_1[i]:
        final_preds.append(2)
    elif preds_1[i] == preds_2[i]:
        final_preds.append(0)

sub_df = pd.DataFrame(columns=["id", "pred"])
sub_df["id"] = total_data_df["id"]
sub_df["pred"] = final_preds
assert len(sub_df) == len(total_data_df)

if LOAD_LM:
    fname = "without_LM" + model_name + "_task2"
else:
    fname = model_name + "_task2"
sub_df.to_csv(SAVE_PATH + fname + "_test.csv", index=False)

