import pickle
import torch
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset
import os
from src.utils import load_config
from collections import defaultdict


def train_tokenizer():
    _, model_config = load_config("./config.json")
    spm.SentencePieceTrainer.Train(input=[f"./data/{file}" for file in os.listdir("./data") if file.endswith('.txt') and "code" in file],
                                   model_prefix='./tokenizer/tokenizer', vocab_size=model_config["vocab_size"],
                                   user_defined_symbols=["<s>", "</s>", "<n>", "<pad>"])
    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')

    target = """import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
"""
    print(sp.Encode(target))
    print(sp.Decode(sp.Encode(target)))


def preprocess_line(line: str):
    space_before_punc = [" ,", " .", " '", " ?", " !", " ;"]
    stuff_to_get_rid = ["``", "''"]
    new_line = line

    for stuff in stuff_to_get_rid:
        new_line = new_line.replace(stuff, "")

    for punc in space_before_punc:
        new_line = new_line.replace(punc, punc[1])

    new_line = new_line.replace(" n't", "n't")
    new_line = new_line.replace("  ", " ")
    new_line = new_line.replace(" i ", " I ")
    new_line = new_line.replace(" i'", " I'")
    new_line = new_line.replace(" $ ", " $")
    new_line = new_line.lstrip().capitalize().rstrip()

    return new_line


def process_book_corpus():
    max_lines_per_file = 100_000

    # with open("./data/tinyshakespeare.txt", "r+") as shakespeare:
    #     text = shakespeare.read()
    #     if not text.startswith("<s>"):
    #         shakespeare.write("<s>" + text.replace("\n\n", "</s>\n\n<s>") + "</s>")

    file_id = 0

    for file_name in os.listdir("./data/bookcorpus"):
        data = load_dataset('parquet', data_files=f'./data/bookcorpus/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/bookcorpus_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                file_total["bookcorpus"] += 1

                for line in line_cluster:
                    f.write(f'<s>{preprocess_line(line)}</s>')

            file_id += 1


def tokenize_data():
    print("Tokenizing data")
    training_config, model_config = load_config("./config.json")
    split_ratio = training_config["split_percent"]

    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')

    file_counter = defaultdict(lambda: 0)
    count_file_total()

    if not os.path.exists('./data/tokenized/val/'):
        os.mkdir('./data/tokenized/val/')
    
    if not os.path.exists('./data/tokenized/train/'):
        os.mkdir('./data/tokenized/train/')

    for file in tqdm(os.listdir("./data"), desc="Tokenizing data"):
        if file.endswith('.txt'):
            file_counter[file.split("_")[0]] += 1

            with open(f"./data/{file}", "r", encoding="utf-8") as f:
                tokenized = torch.Tensor(sp.Encode(f.read()))

            if file_counter[file.split("_")[0]] >= int(file_total[file.split("_")[0]] * split_ratio):
                with open(f"./data/tokenized/val/{file.split('.')[0]}_tokenized.p", "wb") as f:
                    pickle.dump(tokenized, f)
            else:
                with open(f"./data/tokenized/train/{file.split('.')[0]}_tokenized.p", "wb") as f:
                    pickle.dump(tokenized, f)


def clean_up():
    file_list = []
    for file_name in os.listdir("./data"):
        if file_name.endswith(".txt") and ("bookcorpus" in file_name or "code_" in file_name):
            file_list.append(file_name)

    for file in file_list:
        os.remove(f"./data/{file}")


def process_c4():
    file_id = 0
    max_lines_per_file = 10_000

    for file_name in os.listdir("./data/c4"):
        if 'parquet' not in file_name:
            continue

        data = load_dataset('parquet', data_files=f'./data/c4/{file_name}', trust_remote_code=True)["train"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/c4_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+", encoding="utf-8") as f:
                file_total["c4"] += 1

                for line in line_cluster['text']:
                    new_line_replaced = line.replace('\n', '<n>')
                    f.write(f"<s>{new_line_replaced}</s>\n")

            file_id += 1


def process_code_corpus():
    file_id = 0
    max_lines_per_file = 10_000

    for file_name in os.listdir("./data/code_search_net"):
        if 'parquet' not in file_name:
            continue

        data = load_dataset('parquet', data_files=f'./data/code_search_net/{file_name}', trust_remote_code=True)["train"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/code_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+", encoding="utf-8") as f:
                file_total["code"] += 1

                repository_name = line_cluster["repository_name"]
                whole_func_string = line_cluster["whole_func_string"]

                for idx in range(len(repository_name)):
                    cur_whole_func_string = whole_func_string[idx]
                    cur_whole_func_string = cur_whole_func_string.replace('\n', '<n>')

                    f.write(f'<s>{cur_whole_func_string}</s>\n')

            file_id += 1

def count_file_total():
    global file_total
    
    for file in os.listdir("./data"):
        if file.endswith('.txt'):
            file_total[file.split("_")[0]] += 1


def preprocessing():
    # process_c4()
    # process_code_corpus()
    # process_book_corpus()
    # train_tokenizer()
    tokenize_data()

    # clean_up()


if __name__ == "__main__":
    file_total = defaultdict(lambda: 0)
    preprocessing()
