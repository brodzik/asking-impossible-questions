import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModelForQuestionAnswering, AutoTokenizer

from .evaluation import *
from .scheduler import LinearWarmupInverseSqrtDecayScheduler


class Trainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["path"])
        self.pad_on_right = self.tokenizer.padding_side == "right"

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def preprocess(self, examples):
        """
        Tokenize contexts and questions, identify answer tokens.
        Source: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
        """

        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=config["model"]["max_length"],
            stride=config["model"]["stride"],
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
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

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
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
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

    def augment(self, df_original):
        if not self.config["augmentation"]["enabled"]:
            return df_original

        df = df_original.copy()
        spacy_nlp = spacy.load("pl_core_news_lg")

        df_back_translate = pd.DataFrame()

        for path in config["dataset"]["back_translate_paths"]:
            df_back_translate = pd.concat([df_back_translate, pd.read_csv(path).fillna("")])

        for i, row in tqdm(df.iterrows(), total=len(df), desc="augment"):
            context = row["context"]
            question = row["question"]
            answer_text = row["answer_text"].strip()
            answer_start = row["answer_start"]

            context_l = context[:answer_start].strip()
            context_r = context[answer_start + len(answer_text):].strip()

            if config["augmentation"]["p_back_translate"] > 0:
                if np.random.rand() < config["augmentation"]["p_back_translate"]:
                    temp = df_back_translate[df_back_translate["qa_id"] == row["qa_id"]]

                    if len(temp) > 0:
                        temp = temp.sample(1).iloc[0]

                        if np.random.rand() < 0.5:
                            context = temp["context"]
                            answer_text = temp["answer_text"]
                            answer_start = temp["answer_start"]

                        if np.random.rand() < 0.5:
                            question = temp["question"]

            if config["augmentation"]["p_replace_question"] > 0:
                if np.random.rand() < config["augmentation"]["p_replace_question"]:
                    temp = df_original[df_original["group_id"] != row["group_id"]]

                    if len(temp) > 0:
                        question = temp.sample(1).iloc[0]["question"]
                        answer_text = ""
                        answer_start = 0

            if config["augmentation"]["p_replace_question_entity"] > 0:
                if np.random.rand() < config["augmentation"]["p_replace_question_entity"]:
                    temp = df_original[df_original["group_id"] != row["group_id"]]

                    for e in spacy_nlp(question).ents:
                        while True:
                            random_context = temp.sample(1).iloc[0]["context"]
                            random_context_entities = list(set(e.text for e in spacy_nlp(random_context).ents))

                            if len(random_context_entities) > 0:
                                break

                        random_context_entity = np.random.choice(random_context_entities)

                        question = question.replace(e.text, random_context_entity)
                        answer_text = ""
                        answer_start = 0

            if config["augmentation"]["p_drop_answer"] > 0:
                if np.random.rand() < config["augmentation"]["p_drop_answer"]:
                    answer_text = ""
                    answer_start = 0

            if config["augmentation"]["p_drop_token"] > 0:
                context_l = " ".join([t for t in context_l.split() if np.random.rand() > config["augmentation"]["p_drop_token"]])
                context_r = " ".join([t for t in context_r.split() if np.random.rand() > config["augmentation"]["p_drop_token"]])
                answer_text = " ".join([t for t in answer_text.split() if np.random.rand() > config["augmentation"]["p_drop_token"]])

            if config["augmentation"]["p_drop_char"] > 0:
                context_l = "".join([c for c in context_l if np.random.rand() > config["augmentation"]["p_drop_char"]])
                context_r = "".join([c for c in context_r if np.random.rand() > config["augmentation"]["p_drop_char"]])
                answer_text = "".join([c for c in answer_text if np.random.rand() > config["augmentation"]["p_drop_char"]])

            if config["augmentation"]["p_replace_polish_char"] > 0:
                polish_chars = {
                    "ą": "a",
                    "ć": "c",
                    "ę": "e",
                    "ł": "l",
                    "ń": "n",
                    "ó": "o",
                    "ś": "s",
                    "ź": "z",
                    "ż": "z"
                }

                context_l = "".join([(polish_chars[c] if c in polish_chars and np.random.rand() < config["augmentation"]["p_replace_polish_char"] else c) for c in context_l])
                context_r = "".join([(polish_chars[c] if c in polish_chars and np.random.rand() < config["augmentation"]["p_replace_polish_char"] else c) for c in context_r])
                answer_text = "".join([(polish_chars[c] if c in polish_chars and np.random.rand() < config["augmentation"]["p_replace_polish_char"] else c) for c in answer_text])

            answer_text = answer_text.strip()
            context_l = context_l.strip() + " "
            context_r = " " + context_r.strip()
            context = context_l + answer_text + context_r

            df.loc[i, "context"] = context
            df.loc[i, "question"] = question
            df.loc[i, "answer_text"] = answer_text
            df.loc[i, "answer_start"] = len(context_l) if len(answer_text) > 0 else 0
            df.loc[i, "has_answer"] = int(len(answer_text) > 0)

            assert answer_text == df.loc[i, "context"][df.loc[i, "answer_start"]:df.loc[i, "answer_start"] + len(df.loc[i, "answer_text"])]

        print("has_answer", df["has_answer"].sum() / len(df))

        return df

    def get_data_loaders(self, train_df, dev_df):
        train_df = self.augment(train_df)
        train_df["answers"] = train_df[["answer_start", "answer_text"]].apply(lambda x: {"answer_start": [x[0]], "answer_text": [x[1]]}, axis=1)
        train_set = Dataset.from_pandas(train_df)
        train_set = train_set.map(self.preprocess, batched=True, remove_columns=train_set.column_names)
        train_loader = DataLoader(train_set, batch_size=self.config["train"]["batch_size"], shuffle=True)

        dev_df["answers"] = dev_df[["answer_start", "answer_text"]].apply(lambda x: {"answer_start": [x[0]], "answer_text": [x[1]]}, axis=1)
        dev_set = Dataset.from_pandas(dev_df)
        dev_set = dev_set.map(self.preprocess, batched=True, remove_columns=dev_set.column_names)
        dev_loader = DataLoader(dev_set, batch_size=self.config["train"]["batch_size"], shuffle=False)

        return train_loader, dev_loader

    def run(self):
        os.makedirs(self.config["train"]["save_path"], exist_ok=True)

        self.seed_everything(self.config["train"]["seed"])

        df = pd.read_csv(self.config["dataset"]["train_path"]).fillna("").sample(self.config["dataset"]["sample_size"]).reset_index(drop=True)
        df["has_answer"] = df["answer_text"].apply(lambda x: int(len(x) > 0))

        train_df, dev_df = train_test_split(df, test_size=self.config["dataset"]["dev_size"], random_state=self.config["train"]["seed"], shuffle=True, stratify=df["has_answer"])
        del df

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForQuestionAnswering.from_pretrained(self.config["model"]["path"]).to(device=device)
        optimizer = AdamW(model.parameters(), lr=self.config["train"]["lr_initial"], weight_decay=self.config["train"]["weight_decay"])
        scheduler = LinearWarmupInverseSqrtDecayScheduler(optimizer, lr_initial=self.config["train"]["lr_initial"], lr_peak=self.config["train"]["lr_peak"], lr_final=self.config["train"]["lr_final"], t_warmup=self.config["train"]["t_warmup"], t_decay=self.config["train"]["t_decay"])

        pbar = tqdm(range(self.config["train"]["epochs"]))
        best_loss = np.inf

        for epoch in pbar:
            train_loader, dev_loader = self.get_data_loaders(train_df.copy(), dev_df.copy())

            model = model.train()
            batch_pbar = tqdm(train_loader, desc="train", leave=False)

            for i, x in enumerate(batch_pbar):
                model(
                    input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                    attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                    start_positions=x["start_positions"].to(device=device),
                    end_positions=x["end_positions"].to(device=device)
                ).loss.backward()

                if (i + 1) % self.config["train"]["gradient_accumulation"] == 0 or i == len(train_loader) - 1:
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    batch_pbar.set_description("train lr: {}".format(scheduler.get_lr()))

            model = model.eval()
            train_loss = 0
            dev_loss = 0

            with torch.no_grad():
                for x in tqdm(train_loader, desc="eval train", leave=False):
                    train_loss += model(
                        input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                        attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                        start_positions=x["start_positions"].to(device=device),
                        end_positions=x["end_positions"].to(device=device)
                    ).loss.item() / len(train_loader)

                for x in tqdm(dev_loader, desc="eval dev", leave=False):
                    dev_loss += model(
                        input_ids=torch.stack(x["input_ids"], dim=1).to(device=device),
                        attention_mask=torch.stack(x["attention_mask"], dim=1).to(device=device),
                        start_positions=x["start_positions"].to(device=device),
                        end_positions=x["end_positions"].to(device=device)
                    ).loss.item() / len(dev_loader)

            if dev_loss < best_loss:
                best_loss = dev_loss

                if self.config["train"]["save_best"]:
                    model.save_pretrained("{}/best".format(self.config["train"]["save_path"]))
                    self.tokenizer.save_pretrained("{}/best".format(self.config["train"]["save_path"]))

            if self.config["train"]["save_checkpoint"]:
                model.save_pretrained("{}/checkpoint".format(self.config["train"]["save_path"]))
                self.tokenizer.save_pretrained("{}/checkpoint".format(self.config["train"]["save_path"]))

            if self.config["train"]["save_each_epoch"]:
                model.save_pretrained("{}/epoch_{}".format(self.config["train"]["save_path"], epoch))
                self.tokenizer.save_pretrained("{}/epoch_{}".format(self.config["train"]["save_path"], epoch))

            with open("{}/log.txt".format(self.config["train"]["save_path"]), "a") as f:
                f.write("{},{},{}\n".format(epoch, train_loss, dev_loss))

            pbar.set_description("t: {} train: {:.4f} dev: {:.4f} best: {:.4f}".format(scheduler.get_t(), train_loss, dev_loss, best_loss))

        Evaluator(self.config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    Trainer(config).run()
