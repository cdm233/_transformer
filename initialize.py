import torch
from src.model import initialize_model, TransformerNet
from src.utils import *


def ini(dataloader):
    training_config, model_config = load_config("./config.json")

    torch.manual_seed(training_config["manual_seed"])

    initialize_model(model_config, dataloader)
    model = TransformerNet(model_config["vocab_size"]).to(device)

    return model
