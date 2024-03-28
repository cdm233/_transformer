import copy
import time

import torch

from src.model import NoamLR
from src.utils import *
from initialize import ini
from src.model import Generator

model, dataloader = load_local_model("0.4", "stage_best_train")
model.convert_model('fp16')

start_token = dataloader.starting_token
end_token = dataloader.ending_token

# starting_text = input("Input starting text below:")
starting_text = '<s> I have a feeling that you might'

tokenized_text = dataloader.tokenizer.Encode(starting_text)
starting_text = torch.Tensor(tokenized_text).to(device).unsqueeze(dim=0).to(torch.long)

generator = Generator(model, model.model_param["block_size"], dataloader.tokenizer, temperature=0.1, k=1000, p=0.5)

print('Input:', starting_text)

with torch.no_grad():
    model_pred, _ = model(starting_text)

probs = F.softmax(model_pred, dim=-1)
token_predicted = torch.argmax(probs[0][-1])

for token_id in range(len(probs[0])):
    cur_inp_seq = starting_text[0][:token_id + 1]
    cur_inp_text = dataloader.tokenizer.Decode(cur_inp_seq.tolist())

    cur_pred_idx = torch.argmax(probs[0, token_id]).item()
    cur_pred_prob = probs[0, token_id][cur_pred_idx].item()
    cur_pred_token = dataloader.tokenizer.Decode(torch.argmax(probs[0, token_id]).tolist())

    print(f"Given {cur_inp_seq}({cur_inp_text})")
    print(f"Model predicted: {cur_pred_idx}({cur_pred_token}) with probability {cur_pred_prob}")

    try:
        cur_inp_tok = starting_text[0][token_id + 1]
        cur_inp_tok_prob = probs[0, token_id, cur_inp_tok]
    except:
        break

    print(f"The ground truth is {cur_inp_tok}({dataloader.tokenizer.Decode(cur_inp_tok.tolist())}) with probability {cur_inp_tok_prob}")
    print()

print('----------------')
print(model_pred[0][-1], probs[0][-1][token_predicted])
print('token_predicted:', token_predicted)
print(dataloader.tokenizer.Decode(token_predicted.tolist()))
print("\ngenerator.generate: ", generator.generate(starting_text, 30))

# print("generator.random_sample: ", generator.random_sample(starting_text, 50))
# print("\ngenerator.top_p: ", generator.top_p(starting_text, 50))
# print("\ngenerator.beam_search: ", generator.beam_search(starting_text, 50, k=2))
