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
from sklearn.model_selection import train_test_split
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
        self.weights = [1.96, 2.35, 15.32, 15.32]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        grade = torch.tensor(self.df["meanGrade"][idx])
        loss_weight = torch.tensor(self.weights[int(round(self.df["meanGrade"][idx]))])

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


model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

train_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/train.csv")
valid_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/dev.csv")
test_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/test.csv")
test_df["meanGrade"] = [0] * len(test_df)

train_dset = two_sentence_dataset(train_df, tokenizer, max_len=256)
valid_dset = two_sentence_dataset(valid_df, tokenizer, max_len=256)
test_dset = two_sentence_dataset(test_df, tokenizer, max_len=256)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, num_workers=0)

# MODEL
model = transformers.BertForSequenceClassification.from_pretrained(
    model_name, num_labels=1
)
model = model.to(device)

# LOADING LM WEIGHTS
if LOAD_LM:
    fname = model_name + "_LM"
    model.bert.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))

for name, param in model.named_parameters():
    if (
        ("classifier" not in name)
        & ("pooler" not in name)
        & ("layer.11" not in name)
        & ("layer.10" not in name)
    ):
        param.requires_grad = False

optimizer = transformers.AdamW(model.parameters(), lr=1e-4, eps=10e-8)

# TRAINING
if LOAD_LM:
    fname = "without_LM" + model_name + "_task1"
else:
    fname = model_name + "_task1"
train_loss = []
valid_loss = []
min_val_loss = 999999
start_time = time.time()

for epoch in range(5):
    running_loss = 0.0
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        input_ids, token_type_ids, attention_mask, grade = data
        input_ids, token_type_ids, attention_mask, grade = (
            input_ids.to(device),
            token_type_ids.to(device),
            attention_mask.to(device),
            grade.to(device),
        )

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            labels=grade,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 100 == 99:  # print every 25 mini-batches
            print(
                "[%d, %5d] loss: %.5f time: %.3f"
                % (epoch + 1, i + 1, running_loss / 100, time.time() - start_time)
            )
            running_loss = 0.0

    print("\nEPOCH ", epoch + 1, " TRAIN LOSS = ", epoch_loss / len(train_dset))
    train_loss.append(epoch_loss / len(train_dset))

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
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
            loss, logits = outputs[:2]

            val_loss += loss.item()

    print("EPOCH ", epoch + 1, " VALID LOSS = ", val_loss / len(valid_dset), "\n")
    valid_loss.append(val_loss / len(valid_dset))

    if (val_loss / len(valid_dset)) < min_val_loss:
        print("Model Optimized! Saving Weights...", "\n")
        min_val_loss = val_loss / len(valid_dset)
        torch.save(model.state_dict(), SAVE_PATH + fname + ".pt")

# PREDICTIONS ON TEST
preds = []
model.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
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
            preds.append(logit.item())

sub_df = pd.DataFrame(columns=["id", "pred"])
sub_df["id"] = test_df["id"]
sub_df["pred"] = preds
assert len(sub_df) == len(test_df)
sub_df.to_csv(SAVE_PATH + fname + "_test.csv", index=False)
