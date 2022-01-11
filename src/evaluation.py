import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.metric import *


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["train"]["save_path"] + "/best")
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config["train"]["save_path"] + "/best").to(device=self.device).eval()
        self.pad_on_right = self.tokenizer.padding_side == "right"

    def preprocess(self, examples):
        """
        Tokenize contexts and questions.
        Source: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
        """

        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.config["model"]["max_length"],
            stride=self.config["model"]["stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["qa_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["qa_id"].append(examples["qa_id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [(o if sequence_ids[k] == context_index else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

        return tokenized_examples

    def run(self):
        """
        Evaluate model using test set.
        Source: https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb
        """

        df = pd.read_csv(self.config["dataset"]["test_path"]).fillna("").reset_index(drop=True)
        test_set = Dataset.from_pandas(df)
        test_set = test_set.map(self.preprocess, batched=True, remove_columns=test_set.column_names)

        preds = {}
        best_null_score = {}

        for x in tqdm(test_set):
            input_ids = torch.tensor([x["input_ids"]]).to(device=self.device)
            attention_mask = torch.tensor([x["attention_mask"]]).to(device=self.device)

            with torch.no_grad():
                y_pred = self.model(input_ids=input_ids, attention_mask=attention_mask)

            n_best_size = 20
            max_answer_length = 30

            start_logits = y_pred["start_logits"][0].cpu()
            end_logits = y_pred["end_logits"][0].cpu()

            start_indexes = np.argsort(start_logits).tolist()[-1:-n_best_size-1:-1]
            end_indexes = np.argsort(end_logits).tolist()[-1:-n_best_size-1:-1]
            offset_mapping = x["offset_mapping"]

            qa_id = x["qa_id"]
            context = df.loc[df["qa_id"] == qa_id, "context"].iloc[0]

            # Collect all predictions for a given example, since long contexts can be split into multiple parts
            if qa_id not in preds:
                preds[qa_id] = []

            cls_index = x["input_ids"].index(self.tokenizer.cls_token_id)
            null_score = (start_logits[cls_index] + end_logits[cls_index]).numpy().item()

            if qa_id not in best_null_score or null_score > best_null_score[qa_id]:
                best_null_score[qa_id] = null_score

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers
                    if start_index >= len(offset_mapping) or \
                            end_index >= len(offset_mapping) or \
                            offset_mapping[start_index] is None or \
                            offset_mapping[end_index] is None or \
                            end_index < start_index or \
                            end_index - start_index + 1 > max_answer_length:
                        continue

                    # Check if the answer is inside the context
                    if start_index <= end_index:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]

                        preds[qa_id].append({
                            "score": (start_logits[start_index] + end_logits[end_index]).numpy().item(),
                            "text": context[start_char:end_char]
                        })

        f1_all = []
        em_all = []

        f1_answer = []
        em_answer = []

        f1_no_answer = []
        em_no_answer = []

        for k, v in preds.items():
            v.sort(reverse=True, key=lambda x: x["score"])

            gold_text = df.loc[df["qa_id"] == k, "answer_text"].iloc[0]
            if len(v) > 0:
                pred_text = v[0]["text"] if v[0]["score"] > best_null_score[k] else ""
            else:
                pred_text = ""

            f1 = compute_f1(gold_text, pred_text)
            em = compute_em(gold_text, pred_text)

            f1_all.append(f1)
            em_all.append(em)

            if gold_text:
                f1_answer.append(f1)
                em_answer.append(em)
            else:
                f1_no_answer.append(f1)
                em_no_answer.append(em)

        eval_stats = {
            "N": len(f1_all),
            "f1": np.mean(f1_all),
            "em": np.mean(em_all),
            "N_answer": len(f1_answer),
            "f1_answer": np.mean(f1_answer),
            "em_answer": np.mean(em_answer),
            "N_no_answer": len(f1_no_answer),
            "f1_no_answer": np.mean(f1_no_answer),
            "em_no_answer": np.mean(em_no_answer)
        }

        with open(self.config["train"]["save_path"] + "/eval_stats.json", "w") as f:
            json.dump(eval_stats, f, indent=4)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        c = yaml.safe_load(f)

    Evaluator(c).run()
