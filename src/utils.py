import pickle
import pprint
from datetime import datetime, timedelta
import requests
import os
import torch
import csv
import json
import sentencepiece as spm
from tqdm import tqdm
from src.model import initialize_model, TransformerNet
from datasets import load_dataset
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


class CodeDataLoader:
    def __init__(self) -> None:
        # self.dataset = load_dataset("theblackcat102/evol-codealpaca-v1", "en-US", split="train")
        pass


class BaseDataLoader:
    def __init__(self, data_segments=-1, starting_segment=0):
        super(BaseDataLoader, self).__init__()
        sp = spm.SentencePieceProcessor()
        sp.Load('./tokenizer/tokenizer.model')
        self.tokenizer = sp

        self.train_data = None
        self.val_data = None

        if starting_segment < 0:
            starting_segment = 0

        proposed_end = starting_segment + data_segments
        num_train_files = len(os.listdir("./data/tokenized/train"))
        train_data_end = num_train_files if proposed_end == -1 else proposed_end

        for file_name in tqdm(os.listdir("./data/tokenized/train")[starting_segment: train_data_end], desc="Loading training data"):
            with open(f"./data/tokenized/train/{file_name}", "rb") as f:
                if self.train_data is None:
                    self.train_data = pickle.load(f)
                else:
                    self.train_data = torch.concat((self.train_data, pickle.load(f)))
        
        proposed_end = starting_segment + data_segments
        num_train_files = len(os.listdir("./data/tokenized/train"))
        val_data_end = num_train_files if proposed_end == -1 else proposed_end

        for file_name in tqdm(os.listdir("./data/tokenized/val")[starting_segment:val_data_end], desc="Loading val data"):
            with open(f"./data/tokenized/val/{file_name}", "rb") as f:
                if self.val_data is None:
                    self.val_data = pickle.load(f)
                else:
                    self.val_data = torch.concat((self.val_data, pickle.load(f)))

        self.train_data = self.train_data.to(torch.long)
        self.val_data = self.val_data.to(torch.long)

        self.starting_token = 1
        self.ending_token = 2

        print(f"Data loaded. There is {len(self.train_data) / 1e9} B tokens in train data, and "
              f"{len(self.val_data) / 1e9} B tokens in val data.")

    def get_batch(self, batch_size, block_size, train=True):
        if train:
            data = self.train_data
        else:
            data = self.val_data

        rand_idx = torch.randint(len(data) - block_size, (batch_size,))
        inputs = torch.stack([data[i:i + block_size] for i in rand_idx])
        targets = torch.stack([data[i + 1:i + block_size + 1] for i in rand_idx])

        return inputs.to(device), targets.to(device)


class ShakespeareDataLoader:
    def __init__(self, split_percent=0.5, path="./data/tinyshakespeare.txt"):
        self.path = path
        self.text_data = self.get_data()

        sp = spm.SentencePieceProcessor()
        sp.Load('./tokenizer/tokenizer.model')

        # self.tokenizer = Tokenizer(self.text_data)
        self.tokenizer = sp
        tokenized_data = self.tokenizer.Encode(self.text_data)

        self.tokenized_data = torch.tensor(tokenized_data, dtype=torch.long)
        split_index = int(split_percent * len(self.tokenized_data))

        self.starting_token = 1
        self.ending_token = 2

        self.train_data = self.tokenized_data[:split_index]
        self.val_data = self.tokenized_data[split_index:]

    @staticmethod
    def get_data():
        file_exists = os.path.exists("./data/tinyshakespeare.txt")

        if not file_exists:
            data_res = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
            with open("./data/tinyshakespeare.txt", "w") as data_file:
                data_file.write(data_res.text)

            data = data_res.text
        else:
            with open("./data/tinyshakespeare.txt", "r") as data_file:
                data = data_file.read()

        return data

    def get_batch(self, batch_size, block_size, train=True):
        if train:
            data = self.train_data
        else:
            data = self.val_data

        rand_idx = torch.randint(len(data) - block_size, (batch_size,))
        inputs = torch.stack([data[i:i + block_size] for i in rand_idx])
        targets = torch.stack([data[i + 1:i + block_size + 1] for i in rand_idx])

        return inputs.to(device), targets.to(device)


class TrainingLog:
    def __init__(self, model_version, training_name, continued_training=False):
        self.logs = []

        self.cur_log = {
            "Name": training_name,
            "Model Version": model_version,
            "Continuation": continued_training,
            "Creation Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Records": []
        }

        self.continuation = continued_training
        self.model_version = model_version
        self.cur_epoch = 0
        self.name = training_name
        self.parent_dir = f"model_v{self.model_version}"

        try:
            self.load_log()
        except FileNotFoundError:
            self.save_log()

        self.logs.append(self.cur_log)

    def record(self, train_loss, val_loss=None, sample_text=None, epoch=-1):
        temp_log = {
            "cur_epoch": self.cur_epoch,
            "cur_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else "N/A",
            "sample_text": sample_text if sample_text is not None else "N/A"
        }

        if epoch == -1:
            self.cur_log["Records"].append(temp_log)
            self.cur_epoch += 1
        else:
            self.cur_log["Records"][epoch] = temp_log

    def save_log(self):
        with open(f"./models/{self.parent_dir}/{self.name}.p", "wb") as f:
            pickle.dump(self.logs, f)

    def load_log(self, name=None):
        parent_dir = f"model_v{self.model_version}"
        with open(f"./models/{parent_dir}/{name if name is not None else self.name}.p", "rb") as f:
            self.logs = pickle.load(f)


class DataRecorder:
    """
    Provides functionalities to initiates, insert, edit, delete, clear, and print a target csv files.
    """

    def __init__(self, record_dir, headers, file_name="Record_file.csv"):
        """
        Instantiates the recorder, if the recorder file already exists, then the headers param is not used. Otherwise, it is used for initialization.
        """
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        self.parent_dir = record_dir
        self.file_name = file_name
        self.path = os.path.join(self.parent_dir, self.file_name)

        if not os.path.exists(self.path):
            with open(self.path, "w+") as f:
                f.write("")

            self.headers = ['index', 'timestamp'] + headers
            self._update_headers(self.headers)

        else:
            with open(self.path, 'r', newline='') as csvfile:
                if os.stat(self.path).st_size == 0:
                    # If the file somehow is empty and there is no headers in it, create new headers.
                    self.headers = ['index', 'timestamp'] + headers
                    self._update_headers(self.headers)
                else:
                    reader = csv.reader(csvfile)
                    self.headers = next(reader)  # Read the first line to get the headers

    def record_new(self, data):
        new_keys = set(data.keys()) - set(self.headers)
        if new_keys:
            self._update_headers(new_keys)

        # Prepare complete data with placeholders for missing fields
        complete_data = {header: data.get(header, '') for header in self.headers if header in ['index', 'timestamp'] or header in data}

        with open(self.path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            data_with_meta = self._add_meta_data(complete_data)
            writer.writerow(data_with_meta)

    def _add_meta_data(self, data):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, 'r', newline='') as file:
            last_index = sum(1 for _ in file)
        data['index'] = last_index
        data['timestamp'] = timestamp
        return data

    def _update_headers(self, new_keys):
        # Add new keys at the end of the current headers
        original_headers = self.headers.copy()
        self.headers = [header for header in self.headers if header not in new_keys] + list(new_keys)

        updated_data = []
        with open(self.path, 'r', newline='') as file:
            reader = csv.DictReader(file, fieldnames=original_headers)
            headers_row = True
            for row in reader:
                # Skip header row, otherwise we have duplicates
                if headers_row:
                    headers_row = False
                    continue

                for key in new_keys:
                    row[key] = row.get(key, '')  # Set empty string for new key if not present
                updated_data.append(row)

        with open(self.path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(updated_data)

    def _read_all_raw(self):
        """
        Reads the csv file in raw format, used internally by other methods.
        """
        with open(self.path, 'r', newline='') as file:
            reader = csv.reader(file)
            return list(reader)

    def read_all(self):
        """
        User should call this for formatted data.
        """
        with open(self.path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    def edit_row(self, row_index, new_data):
        rows = self._read_all_raw()
        if 0 <= row_index < len(rows):
            rows[row_index] = [new_data.get(field, rows[row_index][i])
                               for i, field in enumerate(rows[0])]
            self._write_all(rows)

    def delete_row(self, row_index):
        rows = self._read_all_raw()
        if 0 <= row_index < len(rows):
            del rows[row_index]
            self._write_all(rows)

    def _write_all(self, rows):
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def clear_file(self):
        with open(self.path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()

    def print_all(self):
        pprint.pprint(self.read_all())


def load_dict(path):
    with open(path, "r") as f:
        data = json.load(f)
        return data


def save_dict(data, path):
    with open(path, "r") as f:
        json.dump(data, f)


def load_config(path="./config.json"):
    config = load_dict(path)

    training_config = config["training"]
    model_config = config["model"]

    return training_config, model_config


def load_local_model(version, model_name):
    path = f"./models/model_v{version}"

    config = load_dict(f"{path}/config.json")

    training_config = config["training"]
    model_config = config["model"]

    dataloader = StreamDataLoader()

    torch.manual_seed(training_config["manual_seed"])

    initialize_model(model_config, dataloader)
    model = TransformerNet(model_config["vocab_size"]).to(device)

    model.load_model(model_name)

    return model, dataloader


class StreamDataLoader:
    def __init__(self, train_start_file_id=0, valid_start_file_id=0):
        sp = spm.SentencePieceProcessor()
        sp.Load('./tokenizer/tokenizer.model')
        self.tokenizer = sp

        self.train_data = None
        self.val_data = None

        self.train_files = list(os.listdir("./data/tokenized/train"))
        self.valid_files = list(os.listdir("./data/tokenized/val"))

        self.cur_train_file_id = train_start_file_id
        self.cur_valid_file_id = valid_start_file_id

        self.cur_train_batch_id = 0
        self.cur_valid_batch_id = 0

        with open(f"./data/tokenized/train/{self.train_files[self.cur_train_file_id]}", "rb") as f:
            self.train_data = pickle.load(f)

        with open(f"./data/tokenized/val/{self.valid_files[self.cur_valid_file_id]}", "rb") as f:
            self.val_data = pickle.load(f)

        self.train_data = self.train_data.to(torch.long)
        self.val_data = self.val_data.to(torch.long)

        self.starting_token = 1
        self.ending_token = 2


    def get_batch(self, batch_size, block_size, train=True, increment=0.75):
        if train:
            data = self.train_data
            cur_batch_id = self.cur_train_batch_id
        else:
            data = self.val_data
            cur_batch_id = self.cur_valid_batch_id

        if cur_batch_id == 0:
            start_idx = 0 
        else:
            start_idx = cur_batch_id * int(increment * block_size)

        idxs = [start_idx]

        for temp_batch_id in range(batch_size - 1):
            cur_start_id = (temp_batch_id + 1) * int(block_size * increment) + start_idx
            if cur_start_id > len(data):
                print("[Dataloader]: Stream in the next file")
                print(f"[Dataloader]: Current stream status: train file id: {self.cur_train_file_id}, val file id: {self.cur_valid_file_id}")

                if train:
                    self.cur_train_file_id += 1
                    self.cur_train_batch_id = 0

                    if self.cur_train_file_id > len(self.train_files) - 1:
                        self.cur_train_file_id = 0

                    with open(f"./data/tokenized/train/{self.train_files[self.cur_train_file_id]}", "rb") as f:
                        self.train_data = pickle.load(f)
                else:
                    self.cur_valid_file_id += 1
                    self.cur_valid_batch_id = 0

                    if self.cur_valid_file_id > len(self.valid_files) - 1:
                        self.cur_valid_file_id = 0
                    
                    with open(f"./data/tokenized/val/{self.valid_files[self.cur_valid_file_id]}", "rb") as f:
                        self.val_data = pickle.load(f)
                
                break

            idxs += [cur_start_id]

        if train:
            self.cur_train_batch_id += batch_size * 1
        else:
            self.cur_valid_batch_id += batch_size * 1

        idxs = torch.Tensor(idxs).to(torch.int)

        def pad_tensor(tensor):
            padding = block_size - tensor.size(0)
            
            if padding > 0:
                padding_result = F.pad(tensor, (0, padding), value=4).to(device).long()

                return padding_result

            return tensor.to(device).long()
        
        inputs = torch.stack([pad_tensor(data[i:i + block_size]) for i in idxs])
        targets = torch.stack([pad_tensor(data[i + 1:i + block_size + 1]) for i in idxs])

        return inputs.to(device).long(), targets.to(device).long() 


def convert_to_datetime(time_str):
    # Get the current datetime details
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    current_day = now.day
    
    # Count occurrences of ":" and "-"
    colon_count = time_str.count(":")
    dash_count = time_str.count("-")
    
    if dash_count == 2 and colon_count == 2:
        format_str = "%Y-%m-%d %H:%M:%S"
    elif dash_count == 2 and colon_count == 1:
        format_str = "%Y-%m-%d %H:%M"
    elif dash_count == 1 and colon_count == 1:
        time_str = f"{current_year}-{time_str}"
        format_str = "%Y-%m-%d %H:%M"
    elif dash_count == 1:
        time_str = f"{current_year}-{time_str} 00:00"
        format_str = "%Y-%m-%d %H:%M"
    elif colon_count == 1:
        if " " in time_str:  # "DD HH:MM"
            day, time = time_str.split()
            time_str = f"{current_year}-{current_month:02d}-{day} {time}"
        else:  # "HH:MM"
            time_str = f"{current_year}-{current_month:02d}-{current_day:02d} {time_str}"
        format_str = "%Y-%m-%d %H:%M"
    elif dash_count == 0 and colon_count == 0 and " " in time_str:
        day, hour = time_str.split()
        time_str = f"{current_year}-{current_month:02d}-{day} {hour}:00"
        format_str = "%Y-%m-%d %H:%M"
    else:
        return None  # Return None for unrecognized formats
    
    try:
        return datetime.strptime(time_str, format_str) + timedelta(hours=5)
    except ValueError:
        return None
