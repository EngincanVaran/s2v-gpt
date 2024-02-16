import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from hex_dataset import HexDataset
from model import GPT
from utils import instantiate_configs


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname)
        if log_color:
            record.msg = f"{log_color}{record.msg}\033[0m"
        return logging.Formatter.format(self, record)


def main():
    logging.info(f"Cuda Available: {torch.cuda.is_available()}")
    config = instantiate_configs("configs.yaml")
    logging.debug("Configs Loaded!")
    trace_file_path = "benign.066165f874547a1cfabce372f202b70bc49f048e1d9a3b758b81df8fa549bd70.trace_12bit.txt"

    dataset = HexDataset(
        file_path=trace_file_path,
        block_size=config.block_size
    )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    logging.debug("Data loaded!")

    model = GPT(config)

    # Optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # Training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    logging.debug("Starting training...")

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

    torch.save(model.state_dict(), "s2v-gpt")


if __name__ == "__main__":
    LOG_COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m'  # Red
    }
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )
    # formatter = ColoredFormatter(LOG_FORMAT)
    # logging.getLogger().handlers[0].setFormatter(formatter)

    main()
