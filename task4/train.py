import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ConvoData(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.tokenized_data = []

        kwargs = {
            "max_length": 768,
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_tensors": "pt"
        }

        for idx in range(len(df)):
            label = df.iloc[idx]["label"]
            resp = df.iloc[idx]["conversation"]
            tokens = tokenizer(resp, **kwargs)
            self.tokenized_data.append((tokens, label))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.tokenized_data[item]


def eval_on_test(epoch, df, test_flag, write=True, loss=None):
    model.eval()

    test_data = ConvoData(df, tokenizer)

    test_dataloader = DataLoader(
        test_data,
        batch_size=4,
        shuffle=False
    )

    true = []
    preds = []

    for batch_id, batch in enumerate(test_dataloader):
        input_ids = batch[0]['input_ids'].squeeze(dim=1).to(device)
        attention_mask = batch[0]['attention_mask'].squeeze(dim=1).to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            out = F.log_softmax(out.logits, dim=-1)

            true += batch[1].tolist()
            preds += out.argmax(dim=-1).tolist()

    if test_flag:
        text = "On test Set:"
    else:
        text = "On sampled train Set:"

    print(text)

    if loss:
        text += f"\n            Avg loss: {loss}"
        print("Avg loss:", loss)

    acc = sum([i == j for i, j in zip(preds, true)]) / len(test_data)
    f1 = f1_score(true, preds, average='macro')

    print("Accuracy:", acc)
    print("F1 score:", f1, "\n")

    if write:
        with open(LOG_PATH, "a") as file:
            w = f"""{text}
            Epoch: {epoch + 1}
            Accuracy: {acc},
            F1 score: {f1}\n
            """
            file.write(w)

    return f1


def save_sub(epoch, f1, label_dict):
    print("Saving submission file...")

    model.eval()

    test_df = pd.read_csv("data/test.csv")
    test_df["label"] = -1

    test_data = ConvoData(test_df, tokenizer)

    test_dataloader = DataLoader(
        test_data,
        batch_size=4,
        shuffle=False
    )

    preds = []

    for batch_id, batch in enumerate(test_dataloader):
        input_ids = batch[0]['input_ids'].squeeze(dim=1).to(device)
        attention_mask = batch[0]['attention_mask'].squeeze(dim=1).to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        out = F.log_softmax(out.logits, dim=-1)

        preds += out.argmax(dim=-1).tolist()

    test_df["preds"] = preds

    sub_grouped = test_df.groupby(by=["conversation_id", "conversation_history"]).agg(list)

    label_dict_rev = {v: k for k, v in label_dict.items()}

    final = []

    for row in sub_grouped.iterrows():
        d = {}
        id = row[0][0]
        hist = row[0][1]
        resp = {}
        for t in range(len(row[1]["tutor"])):
            resp[row[1]["tutor"][t]] = {
                "response": row[1]["response"][t],
                "annotation": {
                    "Actionability": label_dict_rev[row[1]["preds"][t]]
                    # "Mistake_Identification": label_dict_rev[row[1]["preds"][t]]
                    # "Tutor_Identification": label_dict_rev[row[1]["preds"][t]]
                    # "Providing_Guidance": label_dict_rev[row[1]["preds"][t]]
                    # "Mistake_Location": label_dict_rev[row[1]["preds"][t]]
                }
            }

        d["conversation_id"] = id
        d["conversation_history"] = hist
        d["tutor_responses"] = resp

        final.append(d)

    json.dump(final, open(f"{epoch}_{MODEL_ABB}_{f1}_sub.json", "w"))

    return preds



if __name__ == "__main__":

    train_df = pd.read_csv("data/train_all_bea.csv")
    test_df = pd.read_csv("data/valid_all_bea.csv")

    label_dict = {t: i for i, t in enumerate(train_df["actionability"].unique())}

    train_df["label"] = train_df["actionability"].apply(lambda x: label_dict[x])
    test_df["label"] = test_df["actionability"].apply(lambda x: label_dict[x])

    model_id = "microsoft/Phi-4-mini-instruct"
    # model_id = "Qwen/Qwen2.5-7B-Instruct"
    # model_id = "unsloth/Llama-3.2-3B-Instruct"
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_id = "unsloth/Llama-3.2-3B"

    quantization_config = BitsAndBytesConfig(load_in_16bit=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(label_dict),
        quantization_config=quantization_config,
        device_map="cuda:2",
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.config.pad_token_id = model.config.eos_token_id

    train_data = ConvoData(train_df, tokenizer)

    train_dataloader = DataLoader(
        train_data,
        batch_size=8,
    )

    MODEL_ABB = model_id.split("/")[1]
    LOG_PATH = f"logs/training_log_{MODEL_ABB}_action.txt"
    NUM_EPOCHS = 15
    ACCUMULATION_STEPS = 1
    STARTING_LR = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=STARTING_LR)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    loss_fn = nn.CrossEntropyLoss()
    f1 = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_id, batch in enumerate(train_dataloader):
            input_ids = batch[0]['input_ids'].squeeze(dim=1).to(device)
            attention_mask = batch[0]['attention_mask'].squeeze(dim=1).to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )

            loss = loss_fn(out.logits, batch[1].to(device))
            loss.backward()

            if (batch_id + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_id + 1) % 25 == 0:
                print(loss.item())

                if f1 >= 0.68:
                    print(batch_id, "/", len(train_dataloader))
                    f1 = eval_on_test(epoch, test_df, test_flag=True, write=False)
                    if f1 >= 0.685:
                        save_sub(epoch + 1, f1, label_dict)

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        eval_on_test(epoch, train_df.sample(n=400), test_flag=False, loss=avg_loss)
        f1 = eval_on_test(epoch, test_df, test_flag=True)
        save_sub(epoch + 1, f1, label_dict)

        # If the number of batches isn't divisible by accumulation_steps,
        # do one final step after the loop ends
        if (batch_id + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        print(f"Epoch {epoch + 1}")
        print(f"Average Training Loss: {avg_loss:.4f}\n")

        # after each epoch
        if f1 >= 0.685:
            save_sub(epoch + 1, f1, label_dict)
