import logging

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from hex_dataset import HexDataset
from model import GPT
from utils import load_configs, log_configs

BASE_PATH = "/home/ubuntu/state2vec/"
TRACES_PATH = BASE_PATH + "data/traces"

@torch.no_grad()
def estimate_loss(model, train_dataloader, validation_dataloader, eval_iters, device):
    out = {}
    model.eval()
    dataloaders = {
        'train': train_dataloader,
        'val': validation_dataloader,
    }
    for split, dataloader in dataloaders.items():
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(dataloader):
            if k >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)  # Ensure data is on the correct device
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def main(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = GPT(configs.MODEL)
    m = model.to(device)

    logging.info(f"# Parameters: {model.get_num_params() / 1e6:.2f}M")

    TRAINING_SET_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/training.set"
    EXP_FILE = open(TRAINING_SET_PATH)
    EXP_TRACES = []
    for trace in EXP_FILE.readlines():
        trace = trace[
                trace.find("doc/") + 4:
                trace.find("/", trace.find("doc/") + 4)
                ] + ".tar.gz"
        EXP_TRACES.append(trace)
    EXP_FILE.close()


    optimizer = Adam(model.parameters(), lr=configs.TRAINING.learning_rate)

    for index, trace in enumerate(EXP_TRACES):
        index += 1

        logging.info(f"Trace {index}/{len(EXP_TRACES)}")
        logging.info(f"Starting for {trace}")
        trace_file_path = TRACES_PATH + "/" + trace

        # Assuming HexDataset is already defined and initialized
        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=configs.MODEL.block_size
        )
        logging.info("Data Loaded!")

        # Define the sizes of the splits
        total_size = len(dataset)
        train_size = int(0.8 * total_size)  # 80% of the dataset for training
        validation_size = total_size - train_size  # The rest for validation

        # logging.info(f"Data Split! {[train_size, validation_size]}")

        # Split the dataset
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

        # Create DataLoaders for both training and validation sets
        train_dataloader = DataLoader(train_dataset, batch_size=configs.TRAINING.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=configs.TRAINING.batch_size, shuffle=False)

        logging.info("Starting Training...")
        val_loss = float("inf")
        with tqdm(total=configs.TRAINING.max_iters, desc="Training") as pbar:
            for iter in range(configs.TRAINING.max_iters):
                total_loss = 0
                model.train()
                if (iter % configs.TRAINING.eval_interval == 0 and iter > 0) or iter == configs.TRAINING.max_iters-1:
                    losses = estimate_loss(model, train_dataloader, validation_dataloader, configs.TRAINING.eval_interval, device)
                    val_loss = f"{losses['val']:.4f}"
                    # logging.info(f"Step {iter}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

                for batch_idx, (Xb, Yb) in enumerate(train_dataloader):
                    Xb, Yb = Xb.to(device), Yb.to(device)
                    logits, loss = model(Xb, Yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                average_loss = total_loss / len(train_dataloader)
                pbar.update(1)
                pbar.set_postfix(loss=average_loss, val_loss=val_loss)
        losses = estimate_loss(model, train_dataloader, validation_dataloader, configs.TRAINING.eval_interval, device)
        logging.info(f"Final: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        break


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )
    logging.info("***** Running Training *****")

    configs = load_configs("configs.yaml")
    log_configs(configs)

    logging.info(f"Cuda Available: {torch.cuda.is_available()}")

    main(configs)

    logging.info("***** End Training *****")