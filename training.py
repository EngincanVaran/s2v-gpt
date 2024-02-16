import logging
from dataclasses import fields

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from hex_dataset import HexDataset
from model import GPT
from utils import instantiate_configs

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

    # change here
    TRAINING_SET_PATH = f"experiments/exp{config.exp_num}/training.set"
    EXP_FILE = open(TRAINING_SET_PATH)
    EXP_TRACES = []
    for trace in EXP_FILE.readlines():
        trace = trace[
                trace.find("doc/") + 4:
                trace.find("/", trace.find("doc/") + 4)
                ] + ".tar.gz"
        EXP_TRACES.append(trace)

    # eliminate done traces

    for trace in EXP_TRACES:
        logging.info(f"Starting for {trace}")
        trace_file_path = TRACES_PATH + "/" + trace

        # trace_file_path = "benign.066165f874547a1cfabce372f202b70bc49f048e1d9a3b758b81df8fa549bd70.trace_12bit.txt"

        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=config.block_size
        )

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        logging.info(f"Data loaded!")

        model = GPT(config)

        # Optimizer and loss function
        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # Training parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        logging.info("Starting training...")

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
            print(f'\t --> Epoch {epoch + 1}/{config.num_epochs}, Final Train Loss: {avg_loss:.4f}')

        logging.info("Training Finished. Saving model...")
        # torch.save(model.state_dict(), "s2v-gpt_model_state.pth")
        # torch.save(model, 's2v-gpt.pth')
        # logging.info("Model Saved. Exiting...")
        break


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
