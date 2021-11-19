import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModelForQuestionAnswering, AutoTokenizer

from .scheduler import LinearWarmupInverseSqrtDecayScheduler


def seed_everything(seed):
    assert type(seed) == int

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(examples, tokenizer, max_length, stride):
    """
    Tokenize contexts and questions, identify answer tokens.
    Source: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
    """

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if not answers["answer_text"][0]:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["answer_text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            else:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

    return tokenized_examples


def main(config):
    os.makedirs(config["train"]["save_path"])

    seed_everything(config["train"]["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])

    df = pd.read_csv(config["dataset"]["path"] + "/train.csv").fillna("").sample(config["dataset"]["sample_size"]).reset_index(drop=True)
    df["has_answer"] = df["answer_text"].apply(lambda x: int(len(x) > 0))
    df["answers"] = df[["answer_start", "answer_text"]].apply(lambda x: {"answer_start": [x[0]], "answer_text": [x[1]]}, axis=1)

    train_df, dev_df = train_test_split(df, test_size=config["dataset"]["dev_size"], random_state=config["train"]["seed"], shuffle=True, stratify=df["has_answer"])

    train_set = Dataset.from_pandas(train_df)
    train_set = train_set.map(lambda x: preprocess(x, tokenizer, config["model"]["max_length"], config["model"]["stride"]), batched=True, remove_columns=train_set.column_names)
    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True)

    dev_set = Dataset.from_pandas(dev_df)
    dev_set = dev_set.map(lambda x: preprocess(x, tokenizer, config["model"]["max_length"], config["model"]["stride"]), batched=True, remove_columns=dev_set.column_names)
    dev_loader = DataLoader(dev_set, batch_size=config["train"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(config["model"]["path"]).to(device=device)
    optimizer = AdamW(model.parameters(), lr=config["train"]["lr_initial"], weight_decay=config["train"]["weight_decay"])
    scheduler = LinearWarmupInverseSqrtDecayScheduler(optimizer, lr_initial=config["train"]["lr_initial"], lr_peak=config["train"]["lr_peak"], lr_final=config["train"]["lr_final"], t_warmup=config["train"]["t_warmup"], t_decay=config["train"]["t_decay"])

    pbar = tqdm(range(config["train"]["epochs"]))
    best_loss = np.inf

    for epoch in pbar:
        model = model.train()
        batch_pbar = tqdm(train_loader, desc="train")

        for i, x in enumerate(batch_pbar):
            model(
                input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                start_positions=x["start_positions"].to(device=device),
                end_positions=x["end_positions"].to(device=device)
            ).loss.backward()

            if (i + 1) % config["train"]["gradient_accumulation"] == 0 or i == len(train_loader) - 1:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                batch_pbar.set_description("train lr: {}".format(scheduler.get_lr()))

        model = model.eval()
        train_loss = 0
        dev_loss = 0

        with torch.no_grad():
            for x in tqdm(train_loader, desc="eval train"):
                train_loss += model(
                    input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                    attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                    start_positions=x["start_positions"].to(device=device),
                    end_positions=x["end_positions"].to(device=device)
                ).loss.item() / len(train_loader)

            for x in tqdm(dev_loader, desc="eval dev"):
                dev_loss += model(
                    input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                    attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                    start_positions=x["start_positions"].to(device=device),
                    end_positions=x["end_positions"].to(device=device)
                ).loss.item() / len(dev_loader)

        if dev_loss < best_loss:
            best_loss = dev_loss

            if config["train"]["save_best"]:
                model.save_pretrained("{}/best".format(config["train"]["save_path"]))

        if config["train"]["save_checkpoint"]:
            model.save_pretrained("{}/checkpoint".format(config["train"]["save_path"]))

        if config["train"]["save_each_epoch"]:
            model.save_pretrained("{}/epoch_{}".format(config["train"]["save_path"], epoch))

        with open("{}/log.txt".format(config["train"]["save_path"]), "a") as f:
            f.write("{},{},{}".format(epoch, train_loss, dev_loss))

        pbar.set_description("train: {:.4f} dev: {:.4f} best: {:.4f}".format(train_loss, dev_loss, best_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
