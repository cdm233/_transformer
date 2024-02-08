import sentencepiece as spm
import torch
from tqdm import tqdm
import os
import pprint
import pickle
from src.utils import StreamDataLoader
import sys

device = 'cuda'


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


if __name__ == "__main__":
    args = list(sys.argv[1:])
    args_pair = [args[i:i + 2] for i in range(0, len(args), 2)]

    print(args_pair)

    dataloader = StreamDataLoader()
    print(dataloader.train_files[:5])

    # data = dataloader.get_batch(1, 256)

    # print(dataloader.tokenizer.Decode(data[0].tolist()))

    # print(dataloader.train_files)