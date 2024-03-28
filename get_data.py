import pickle
import torch
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset
import os
from src.utils import load_config
from collections import defaultdict
import random
import requests


def process_web_text():
    max_lines_per_file = 4_000

    file_id = 0

    for file_name in os.listdir("./data/parquets/webtext"):
        data = load_dataset('parquet', data_files=f'./data/parquets/webtext/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/plain/webtext_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                for line in line_cluster:
                    f.write(f"<s> {line}</s>\n")

            file_id += 1
        break

def clean_up():
    file_list = []
    for file_name in os.listdir("./data/plain"):
        if file_name.endswith(".txt"):
            file_list.append(file_name)

    for file in file_list:
        os.remove(f"./data/plain/{file}")


def process_textbook2():
    max_lines_per_file = 4_000

    file_id = 0

    for file_name in os.listdir("./data/parquets/textbooks2"):
        data = load_dataset('parquet', data_files=f'./data/parquets/textbooks2/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/plain/textbooks2_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                for line in line_cluster:
                    f.write(f"<s> {line}</s>\n")

            file_id += 1
        break


def process_cnndailymail():
    dataset = load_dataset("cnn_dailymail", trust_remote_code=True, split="train")



def get_data():
    dataset_list = [
        ["nampdn-ai/tiny-textbooks", "textbook"],
        ["Locutusque/UltraTextbooks", "text"],
        ["code_search_net", "all", "whole_func_string"],
        ["bookcorpus", "text"],
        ["allenai/c4", "en", "text"],
        ["oscar", "unshuffled_deduplicated_en", "text"],
        ["Skylion007/openwebtext", "text"],
        ["togethercomputer/RedPajama-Data-1T-Sample", "text"],
        ["codeparrot/github-code-clean", "all-all", "code"],
    ]

    # process_textbook2()

    for dataset_config in dataset_list:
        print('loading', dataset_config)
        dataset = load_dataset(*dataset_config[:-1], trust_remote_code=True, split="train")[dataset_config[-1]]

        print("Loaded!\n")
        break

        file_id = 0
        max_lines_per_file = 2_000

        rearranged = [dataset[i:i + max_lines_per_file] for i in range(0, len(dataset), max_lines_per_file)]

        for line_cluster in rearranged:
            result_file_path = f"./data/plain/{dataset_config[0].split('/')[-1]}_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+", encoding="utf-8") as f:
                for line in line_cluster:
                    f.write(f"<s> {line}</s>\n")

            file_id += 1


def train_tokenizer():
    _, model_config = load_config("./config.json")
    
    file_list = [f"./data/plain/{file}" for file in os.listdir("./data/plain") if file.endswith('.txt')]
    random.shuffle(file_list)
    file_list = file_list[:1000]

    spm_options = dict(
        input_format="text",
        # BPE alg
        model_type="bpe",
        # normalization
        normalization_rule_name="identity", # ew, turn off normalization
        remove_extra_whitespaces=False,
        max_sentence_length=4192, # max number of bytes per sentence
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True,
        # merge rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=False,    # Adds a space before data?
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0, # the UNK token MUST exist
        bos_id=1, # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(), # use ~all system resources
    )
    
    spm.SentencePieceTrainer.Train(input=file_list,
                                   model_prefix='./tokenizer/tokenizer', vocab_size=model_config["vocab_size"],
                                   input_sentence_size=300000,
                                   user_defined_symbols=["<s>", "</s>", "<pad>"],
                                   **spm_options)
    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')

    target = """import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        self.a = 1"""
    print(sp.Encode(target))
    print(sp.Decode(sp.Encode(target)))


def post_tokenization(tokenized, sp):
    _, model_config = load_config("./config.json")
    block_size = model_config["block_size"]

    min_token_span = int(2 * block_size)

    bos = torch.tensor([1.0])
    eos = torch.tensor([2.0])
    tokenized = torch.concat([bos, tokenized, eos]).tolist()

    return tokenized


def tokenize_data():
    print("Tokenizing data")
    training_config, model_config = load_config("./config.json")
    split_ratio = training_config["split_percent"]

    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')
    # sp.SetEncodeExtraOptions('eos:bos')

    if not os.path.exists('./data/tokenized/'):
        os.mkdir('./data/tokenized/')

    if not os.path.exists('./data/tokenized/val/'):
        os.mkdir('./data/tokenized/val/')
    
    if not os.path.exists('./data/tokenized/train/'):
        os.mkdir('./data/tokenized/train/')

    file_list = os.listdir("./data/plain")
    random.shuffle(file_list)

    for file_id, file in tqdm(enumerate(file_list), desc="Tokenizing data", total=len(file_list)):
        if file.endswith('.txt'):
            with open(f"./data/plain/{file}", "r", encoding="utf-8") as f:
                tokenized = torch.Tensor(sp.Encode(f.read()))

            if file_id > int(split_ratio * len(file_list)):
                with open(f"./data/tokenized/val/{file.split('.')[0]}_tokenized.p", "wb") as f:
                    pickle.dump(tokenized, f)
            else:
                with open(f"./data/tokenized/train/{file.split('.')[0]}_tokenized.p", "wb") as f:
                    pickle.dump(tokenized, f)

            del tokenized

        break


def test_tokenization():
    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')
    with open(f"./data/tokenized/train/webtext_0016_plain_tokenized.p", "rb") as f:
        tokenized_data = pickle.load(f).to(torch.int).tolist()
        print(tokenized_data[:100])
        print(sp.Decode(tokenized_data[:4096]))


if __name__ == "__main__":
    clean_up()
    get_data()
    # train_tokenizer()
    # tokenize_data()
