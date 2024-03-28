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


def auto_dataset_downloader(name, base_url):
    if not os.path.exists(f"./data/{name}"):
        os.mkdir(f"./data/{name}")

    file_id = 0

    while True:
        target_file_url = f"{'/'.join(base_url.split('/')[:-1])}/{file_id:04d}.parquet"
        print("Trying to download", target_file_url)

        status_code = requests.head(target_file_url).status_code
        
        if status_code < 400:
            print("File exists")
            os.system(f'wget -P ./data/{name} {target_file_url}')
            file_id += 1
        else:
            print(f"There is an error when downloading file: {status_code}, stopping...")
            break


def train_tokenizer():
    _, model_config = load_config("./config.json")
    
    file_list = [f"./data/{file}" for file in os.listdir("./data") if file.endswith('.txt')]
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
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0, # the UNK token MUST exist
        bos_id=1, # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=-1,
        bos=True,  # Add <s> token
        eos=True,  # Add </s> token
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
    _, model_config = load_config("./config.json")

    block_size = model_config['block_size']

    max_lines_per_file = 100_000

    file_id = 0

    for file_name in os.listdir("./data/bookcorpus"):
        data = load_dataset('parquet', data_files=f'./data/bookcorpus/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/bookcorpus_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                file_total["bookcorpus"] += 1

                line_to_write = ""
                for line in line_cluster:
                    if len(line_to_write) < block_size * 2:
                        line_to_write = line_to_write + " " + preprocess_line(line)
                    else:
                        f.write(f'<s>{line_to_write}</s>\n')
                        line_to_write = ""

            file_id += 1


def tokenize_data():
    print("Tokenizing data")
    training_config, model_config = load_config("./config.json")
    split_ratio = training_config["split_percent"]

    sp = spm.SentencePieceProcessor()
    sp.Load('./tokenizer/tokenizer.model')

    file_counter = defaultdict(lambda: 0)
    count_file_total()

    print('file_total', file_total)

    if not os.path.exists('./data/tokenized/'):
        os.mkdir('./data/tokenized/')

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

            del tokenized


def clean_up():
    file_list = []
    for file_name in os.listdir("./data"):
        if file_name.endswith(".txt"):
            file_list.append(file_name)

    for file in file_list:
        os.remove(f"./data/{file}")


def process_oscar():
    file_id = 0
    max_lines_per_file = 2_000

    for file_name in os.listdir("./data/oscar"):
        if 'parquet' not in file_name:
            continue

        data = load_dataset('parquet', data_files=f'./data/oscar/{file_name}', trust_remote_code=True)["train"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/oscar_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+", encoding="utf-8") as f:
                file_total["oscar"] += 1

                for line in line_cluster['text']:
                    f.write(f"<s>{line}</s>\n")

            file_id += 1


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
                    f.write(f"<s>{line}</s>\n")

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

                    f.write(f'<s>{cur_whole_func_string}</s>\n')

            file_id += 1

def count_file_total():
    global file_total
    
    for file in os.listdir("./data"):
        if file.endswith('.txt'):
            file_total[file.split("_")[0]] += 1


def process_web_text():
    max_lines_per_file = 4_000

    file_id = 0

    for file_name in os.listdir("./data/webtext"):
        data = load_dataset('parquet', data_files=f'./data/webtext/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/webtext_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                file_total["webtext"] += 1

                for line in line_cluster:
                    f.write(f'<s>{line}</s>\n')

            file_id += 1


def process_text_books():
    max_lines_per_file = 4_000

    file_id = 0

    for file_name in os.listdir("./data/textbooks"):
        data = load_dataset('parquet', data_files=f'./data/textbooks/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/textbooks_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                file_total["textbooks"] += 1

                for line in line_cluster:
                    f.write(f'<s>{line}</s>\n')

            file_id += 1


def process_generic_data(dir_name):
    max_lines_per_file = 4_000

    file_id = 0

    for file_name in os.listdir(f"./data/{dir_name}"):
        data = load_dataset('parquet', data_files=f'./data/{dir_name}/{file_name}', trust_remote_code=True)["train"]["text"]

        rearranged = [data[i:i + max_lines_per_file] for i in range(0, len(data), max_lines_per_file)]

        for line_cluster in tqdm(rearranged, desc=f"Processing {file_name}", unit="cluster"):
            result_file_path = f"./data/{dir_name}_{file_id:04d}_plain.txt"

            with open(result_file_path, "w+") as f:
                file_total[dir_name] += 1

                for line in line_cluster:
                    f.write(f'<s>{line}</s>\n')

            file_id += 1


def process_all_generic_data(dir_list):
    for dir_name in os.listdir('./data'):
        if 'tokenized' in dir_name:
            continue

        if dir_name not in dir_list:
            continue
        
        if os.path.isdir(f"./data/{dir_name}"):
            print(dir_name)
            try:
                process_generic_data(dir_name)
            except:
                print(f"Failed process {dir_name}")


def preprocessing():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
    # process_oscar()
    process_all_generic_data(["tinytextbook", "oscar"])
    process_text_books()
    # process_c4()
    # process_code_corpus()
    # process_book_corpus()
    process_web_text()
    # train_tokenizer()
    tokenize_data()

    clean_up()


def download_data():
    # auto_dataset_downloader('oscar',              r'https://huggingface.co/datasets/oscar/resolve/refs%2Fconvert%2Fparquet/unshuffled_deduplicated_en/partial-train/0000.parquet')
    # auto_dataset_downloader('tinytextbook',       r'https://huggingface.co/datasets/nampdn-ai/tiny-textbooks/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet')
    # auto_dataset_downloader('textbooks2',         r'https://huggingface.co/datasets/Locutusque/UltraTextbooks-2.0/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet')
    # auto_dataset_downloader('code_search_net',    r'https://huggingface.co/datasets/code_search_net/resolve/refs%2Fconvert%2Fparquet/python/train/0000.parquet')
    # auto_dataset_downloader('bookcorpus',         r'https://huggingface.co/datasets/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet')
    # auto_dataset_downloader('textbooks',          r'https://huggingface.co/datasets/Locutusque/UltraTextbooks/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet')
    # auto_dataset_downloader('c4',                 r'https://huggingface.co/datasets/allenai/c4/resolve/refs%2Fconvert%2Fparquet/en/partial-train/0000.parquet')

    # auto_dataset_downloader('webtext', r'https://huggingface.co/datasets/oscar/resolve/refs%2Fconvert%2Fparquet/unshuffled_deduplicated_en/partial-train/0000.parquet')

    # /oscar
    # /nampdn-ai/tiny-textbooks
    # /Locutusque/UltraTextbooks
    # /code_search_net
    # /bookcorpus
    # /Locutusque/UltraTextbooks
    # /allenai/c4
    pass


def test_loading():
    from datasets import load_dataset
    dataset = load_dataset('nampdn-ai/tiny-textbooks')["train"]["textbook"]

    max_lines_per_file = 4_000

    rearranged = [dataset[i:i + max_lines_per_file] for i in range(0, len(dataset), max_lines_per_file)]

    for line_cluster in rearranged:
        for line in line_cluster:
            print(line)
            break
        break


if __name__ == "__main__":
    file_total = defaultdict(lambda: 0)
    # test_loading()
    download_data()
    # process_all_generic_data()
    # preprocessing()
    # clean_up()
    tokenize_data()
