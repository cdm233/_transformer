import copy
import time

import torch

from src.model import NoamLR
from src.utils import *
from initialize import ini
from src.model import Generator

model, dataloader = load_local_model("0.28", "stage_best_train")

start_token = dataloader.starting_token
end_token = dataloader.ending_token

starting_text = input("Input starting text below:")
starting_text = starting_text.replace("\n", "<n>")

tokenized_text = dataloader.tokenizer.Encode(starting_text)
starting_text = torch.Tensor(tokenized_text).to(device).unsqueeze(dim=0).to(torch.long)

generator = Generator(model, model.model_param["block_size"], dataloader.tokenizer, temperature=0.9, k=1000, p=0.5)

# print("generator.random_sample: ", generator.random_sample(starting_text, 50))
# print("\ngenerator.top_p: ", generator.top_p(starting_text, 50))
# print("\ngenerator.beam_search: ", generator.beam_search(starting_text, 50, k=2))
print("\ngenerator.generate: ", generator.generate(starting_text, 50))
