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

# SEED
seed_val = 1234
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# DEVICE
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using ", device)

# DATA
train_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/train.csv")
valid_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/dev.csv")
test_df = pd.read_csv("semeval-2020-task-7-data-full/task-1/test.csv")


def get_sentence(sent_orig, edit_word):
    sent_e = (sent_orig.split("<"))[0] + edit_word + (sent_orig.split(">"))[1]
    return sent_e


# GETTING SENTENCE LIST
folders = ["semeval-2020-task-7-data-full/"]
files = []
sentence_list = []
for folder in folders:
    for root, directories, filenames in os.walk(folder):
        for filename in filenames:
            files.append(os.path.join(root, filename))
files = [f for f in files if ((".csv" in f) & ("task-1" in f))]

for f in files:
    df = pd.read_csv(f)
    df.dropna(inplace=True)
    for i in range(len(df)):
        sentence_list.append(get_sentence(df["original"][i], df["edit"][i]))


class lm_dataset(Dataset):
    def __init__(self, sentence_list, tokenizer, max_len=256):
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.sentence_list[idx]
        input_ids = self.tokenizer.encode(
            sentence, add_special_tokens=True, max_length=self.max_len
        )

        input_tensor = torch.tensor(input_ids)
        attention_mask = torch.tensor(
            [1] * len(input_tensor) + [0] * (self.max_len - len(input_tensor))
        )

        if len(input_tensor) < self.max_len:
            input_tensor = torch.cat(
                (
                    input_tensor,
                    (torch.ones(self.max_len - len(input_tensor)) * self.pad).long(),
                )
            )
        elif len(input_tensor) > self.max_len:
            input_tensor = input_tensor[: self.max_len]

        return input_tensor, attention_mask


tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
dataset = lm_dataset(sentence_list, tokenizer, max_len=256)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# MODEL
model_name = "bert-base-uncased"
model = transformers.BertForMaskedLM.from_pretrained(model_name)
model = model.to(device)
optimizer = transformers.AdamW(
    model.parameters(), lr=1e-5, eps=10e-8, weight_decay=1e-3
)

fname = model_name + "_LM"
train_loss = []
start_time = time.time()
for epoch in range(3):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(dataloader):
        input_ids, attention_mask = data
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            masked_lm_labels=input_ids,
        )
        loss, _ = outputs[:2]

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        train_loss.append(loss.item())

        if i % 50 == 49:  # print every 25 mini-batches
            print(
                "[%d, %5d] loss: %.5f time: %.3f"
                % (epoch + 1, i + 1, running_loss / 50, time.time() - start_time)
            )
            running_loss = 0.0

    print("\nEPOCH ", epoch + 1, " TRAIN LOSS = ", epoch_loss / len(dataset), "\n")
    torch.save(model.roberta.state_dict(), SAVE_PATH + fname + ".pt")

# PLOTS
fig = plt.figure()
plt.plot(train_loss, label="Train Loss")

plt.legend()
plt.show()
fig.savefig(SAVE_PATH + fname + "loss.png", dpi=400)

