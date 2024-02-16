import json
import logging
import os
from dataclasses import fields

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from hex_dataset import HexDataset
from model import GPT
from utils import instantiate_configs, send_mail

BASE_PATH = "/home/ubuntu/state2vec/"
TRACES_PATH = BASE_PATH + "data/traces"


def main():
    logging.info(f"Cuda Available: {torch.cuda.is_available()}")
    config = instantiate_configs("configs.yaml")
    for field in fields(config):
        field_name = field.name
        field_value = getattr(config, field_name)
        logging.info(f"{field_name}: {field_value}")
    logging.info("Configs Loaded!")

    if os.path.exists('./models/s2v-gpt_latest.pth'):
        model = torch.load("./models/s2v-gpt_latest.pth")
    else:
        model = GPT(config)
        # Optimizer and loss function
        torch.save(model, './models/s2v-gpt_base.pth')
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    DETAILS_JSON_PATH = f"experiments/exp{config.exp_num}/details.json"
    DETAILS_JSON_FILE = open(DETAILS_JSON_PATH, "r")
    DETAILS_JSON = json.load(DETAILS_JSON_FILE)
    DETAILS_JSON_FILE.close()

    TRAINING_SET_PATH = f"experiments/exp{config.exp_num}/training.set"
    EXP_FILE = open(TRAINING_SET_PATH)
    EXP_TRACES = []
    for trace in EXP_FILE.readlines():
        trace = trace[
                trace.find("doc/") + 4:
                trace.find("/", trace.find("doc/") + 4)
                ] + ".tar.gz"
        EXP_TRACES.append(trace)
    EXP_FILE.close()

    for index, trace in enumerate(EXP_TRACES):
        index += 1
        if DETAILS_JSON[trace]:
            logging.info(f"Model trained with {trace}, skipping...")
            continue

        logging.info(f"Trace {index}/{len(EXP_TRACES)}")
        logging.info(f"Starting for {trace}")
        trace_file_path = TRACES_PATH + "/" + trace

        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=config.block_size
        )

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        logging.info(f"Data Loaded!")

        logging.info("Starting training...")
        avg_loss = 0
        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0

            # Create a tqdm progress bar for batches with dynamic_ncols=True
            batch_progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{config.num_epochs}",
                leave=False,
                dynamic_ncols=True,
                # colour="\033[92m",
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{percentage:3.0f}%] eta: {remaining} {postfix}"
            )

            for X_batch, Y_batch in batch_progress:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                # Forward pass: Compute predicted Y by passing X to the model
                logits, loss = model(X_batch, Y_batch)
                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate and display training loss for each batch
                batch_loss = loss.item()
                batch_progress.set_postfix({'train_loss': batch_loss}, refresh=True)

            avg_loss = total_loss / len(dataloader)
            logging.info(f'\t --> Epoch {epoch + 1}/{config.num_epochs}, Final Train Loss: {avg_loss:.4f}')

        logging.info("Training Finished! Saving model...")
        torch.save(model, f'./models/s2v-gpt_{index}.pth')
        torch.save(model, './models/s2v-gpt_latest.pth')
        logging.info("Model Saved!")
        DETAILS_JSON[trace] = True
        with open("./experiments/exp1/details.json", "w") as f:
            json.dump(DETAILS_JSON, f)
        send_mail(f"Model trained with {trace}. \nTraining Loss:{avg_loss}")
    send_mail(f"Model training done!")

    with open("./experiments/exp1/details.json", "w") as f:
        json.dump(DETAILS_JSON, f)


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )

    main()
