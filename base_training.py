import copy
import time
import sys
from src.model import NoamLR, Generator
from src.utils import *
from initialize import ini
from tqdm import tqdm
import bitsandbytes as bnb
import traceback


def train(total_epochs: list):
    eval_interval = 200
    logger_save_interval = 1

    start_token = dataloader.starting_token
    end_token = dataloader.ending_token

    batch_size = training_config["batch_size"]
    gradient_acc_steps = training_config["gradient_acc"]

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    best_val_model = None
    best_train_model = None

    last_val_loss = 0

    rest_interval = 5000

    model.train()

    for epoch in (bar := tqdm(range(total_epochs[0]), desc=f"Training, gradient acc={gradient_acc_steps}", unit="epoch")):
        if global_stop_time is not None:
            if datetime.now() > global_stop_time:
                print("Global stop time hit, stopping...\n")
                break

        total_epochs[0] -= 1
        model.zero_grad()  # Initialize gradients

        for step in range(gradient_acc_steps):
            if (epoch + 1) % rest_interval == 0:
                # Just to let my GPU rest a bit
                time.sleep(200)
            inputs, targets = dataloader.get_batch(batch_size, model_config["block_size"])

            logits, loss = model(inputs, targets)
            loss = loss / gradient_acc_steps  # Normalize the loss
            loss.backward()

            if (step + 1) % gradient_acc_steps == 0:
                # Perform optimization step after accumulating gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()  # Update the learning rate
                model.zero_grad()  # Reset gradients

            bar.desc = f"Cur training loss: {(loss.item() * gradient_acc_steps):.5f}, val loss: {last_val_loss}"

            if epoch % logger_save_interval == 0:
                logger.record(train_loss=loss.item())

        loss *= gradient_acc_steps

        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            best_train_model = copy.deepcopy(model)

        if (epoch + 1) % eval_interval == 0:
            sample_text = None
            if (epoch + 1) % (eval_interval * 2) == 0:
                starting_text = torch.zeros((1, 1), dtype=torch.long).to(device)
                starting_text[0, 0] = start_token
                print(f"Sample text at epoch{epoch}, train loss: {loss.item()}")
                sample_text = generator.generate(starting_text=starting_text, max_new_tokens=100, end_token=None)
                print(sample_text)
                print()

            estimated_loss = model.estimate_loss()
            last_val_loss = estimated_loss

            logger.record(train_loss=loss.item(), val_loss=float(estimated_loss), sample_text=sample_text, epoch=epoch)

            if float(estimated_loss) < best_val_loss:
                best_val_loss = float(estimated_loss)
                best_val_model = copy.deepcopy(model)

    logger.save_log()

    try:
        # best_train_model.save_model(f"stage_best_train_loss={best_train_loss:.5f}")
        # best_val_model.save_model(f"stage_best_val_loss={best_val_loss:.5f}")
        best_train_model.save_model(f"stage_best_train")
        best_val_model.save_model(f"stage_best_val")

    except AttributeError:
        # Just ignore it, can only hit in debugging
        pass

    starting_text = torch.zeros((1, 1), dtype=torch.long).to(device)
    starting_text[0, 0] = start_token

    print("Cur stage sample text:\n", generator.generate(starting_text=starting_text, max_new_tokens=100, end_token=end_token))


if __name__ == "__main__":
    args = list(sys.argv[1:])

    args_pair = [args[i:i + 2] for i in range(0, len(args), 2)]

    training_config, model_config = load_config("./config.json")

    gradient_acc_steps = training_config["gradient_acc"]

    if training_config["manual_seed"] != -1:
        torch.manual_seed(training_config["manual_seed"])

    dataloader = StreamDataLoader(train_start_file_id=511, valid_start_file_id=47)
    model = ini(dataloader)

    generator = Generator(model, model.model_param["block_size"], dataloader.tokenizer, temperature=1, k=1000, p=0.5)

    model.load_model("stage_best_train")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["lr"], weight_decay=training_config["weight_decay"])

    # optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=training_config["lr"], weight_decay=training_config["weight_decay"],
                                        #  min_8bit_size=16384)

    model_size = model_config["model_arch"]["embedding"]["embed_dim"]
    warmup_steps = 5000  # This is a hyperparameter you can tune
    scheduler = NoamLR(optimizer, model_size, warmup_steps)

    logger = TrainingLog(model_version=model.model_version, training_name="Model Training")

    print("Model has ", model.get_model_params() / 1e6, "M parameters")
    # pprint.pprint(model)

    time.sleep(0.1)

    epochs_to_train = [training_config["total_epoch"]]

    global_stop_time = None

    for pair in args_pair:
        if pair[0] == '-e':
            epochs_to_train = [int(pair[1])]

        if pair[0] == '-t':
            time_str = pair[1]

            time_obj = convert_to_datetime(time_str)

            if time_obj:
                global_stop_time = time_obj
                print(f"Stopping at {global_stop_time}")
            else:
                print("Time string not recognized.")

    while True:
        try:
            train(epochs_to_train)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, pausing...")
            # time.sleep(1)

            sys.stdin.flush()
            flag = input("Interrupted, continue or request text generation? (y/n/<s>)")

            if flag == "n":
                print(f"Current data status: train file id: {dataloader.cur_train_file_id}, val file id: {dataloader.cur_valid_file_id}")
                print("Exiting...")
                model.save_model("interrupt_save", estimate_loss=False)
                logger.save_log()

                break
            elif flag == "<s>":
                model.save_model("just_in_case_save", estimate_loss=False)
                logger.save_log()

                start_text = input("Requesting text generation, please enter the starting text below:")
                start_text = f"<s>{start_text}"
                start_text = torch.Tensor(dataloader.tokenizer.Encode(start_text)).unsqueeze(0).long().to(device)

                print("generated text: \n", generator.generate(start_text, max_new_tokens=200, end_token=None))
            else:
                train(epochs_to_train)
        except:
            model.save_model("error_save", estimate_loss=False)
            logger.save_log()
            
            print("There was an error:")
            print(traceback.format_exc())

            break

        global_stop_time = None

        more_epochs = int(input("\nTraining finished, train for more epochs? Please enter an integer(-1 to stop training):"))

        if more_epochs != -1:
            print(f"Training for {more_epochs} epochs more")
            epochs_to_train = [more_epochs]
        else:
            print("Exiting...")
            break
