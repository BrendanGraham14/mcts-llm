"""

Utilities to load various datasets.

Dataset dfs are expected to have the following columns:
- question
- answer

"""

import pandas as pd

from datasets import load_dataset
import json


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_aime(year: int | None = None) -> pd.DataFrame:
    df = pd.read_csv("datasets/AIME_Dataset_1983_2024.csv")
    df.rename(columns={"Question": "question", "Answer": "answer"}, inplace=True)
    if year is not None:
        df = df[df["Year"] == year]

    return df


def load_gs8mk(year: int | None = None) -> pd.DataFrame:
    data = load_jsonl("datasets/gs8mk-test.jsonl")
    return pd.DataFrame(data)


def load_gsm_hard() -> pd.DataFrame:
    dataset = load_dataset("reasoning-machines/gsm-hard")
    data = [
        {
            "question": dataset["train"][i]["input"],
            "code": dataset["train"][i]["code"],
            "answer": dataset["train"][i]["target"],
        }
        for i in range(len(dataset["train"]))
    ]
    return pd.DataFrame(data)
