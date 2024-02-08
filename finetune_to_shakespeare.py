import time

from tqdm import tqdm
from src.utils import *
from src.model import initialize_model, TransformerNet, NoamLR


def main():
    eval_interval = 100
    start_token = dataloader.starting_token
    end_token = dataloader.ending_token

    best_model_dict = model.state_dict().copy()

    batch_size = training_config["batch_size"]

    best_val_loss = float("inf")

    total_epochs = training_config["total_epoch"]

    last_val_loss = 0
    loss = None

    for epoch in (bar := tqdm(range(total_epochs), desc="Training", unit="epoch")):
        inputs, targets = dataloader.get_batch(batch_size, model_config["block_size"])

        logits, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()  # Update the learning rate

        bar.desc = f"Cur training loss: {loss.item():.5f}, val loss: {last_val_loss}"

        if (epoch + 1) % eval_interval == 0:
            if (epoch + 1) % (eval_interval * 5) == 0:
                starting_text = torch.zeros((1, 1), dtype=torch.long).to(device)
                starting_text[0, 0] = start_token
                print(f"Sample text at epoch{epoch}, train loss: {loss.item()}")
                print(dataloader.tokenizer.decode(model.generate(idx=starting_text, max_new_tokens=500, end_token=end_token)[0].tolist()))
                print()

            estimated_loss = model.estimate_loss()
            last_val_loss = estimated_loss

            if float(estimated_loss) < best_val_loss:
                best_val_loss = float(estimated_loss)
                if epoch / total_epochs > 0.1:
                    best_model_dict = model.state_dict().copy()

    model.save_current_checkpoint(-1, loss.item())

    model.load_state_dict(best_model_dict)

    final_loss = model.estimate_loss()
    model.save_current_checkpoint(-1, loss=final_loss)
    model.eval()

    print("Final loss:", final_loss)

    print("Sample text:")
    starting_text = torch.zeros((1, 1), dtype=torch.long).to(device)
    starting_text[0, 0] = start_token

    print(dataloader.tokenizer.decode(model.generate(idx=starting_text, max_new_tokens=500, end_token=end_token)[0].tolist()))


if __name__ == "__main__":
    training_config, model_config = load_config("./config.json")

    torch.manual_seed(training_config["manual_seed"])

    dataloader = ShakespeareDataLoader(training_config["split_percent"])

    print("---- Initializing Model ----")
    initialize_model(model_config, dataloader)
    model = TransformerNet(model_config["vocab_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"], weight_decay=training_config["weight_decay"])

    model_size = model_config["model_arch"]["embedding"]["embed_dim"]
    warmup_steps = 4000  # This is a hyperparameter you can tune
    scheduler = NoamLR(optimizer, model_size, warmup_steps)

    print("Model has ", model.get_model_params() / 1e6, "M parameters with settings:")
    pprint.pprint(model.model_param)

    print("\n---- Training ----")

    # This makes tqdm happy and not screw with me
    time.sleep(0.1)
    main()
