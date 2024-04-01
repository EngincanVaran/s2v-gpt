import logging
import os.path
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from hex_dataset import HexDataset
from model import GPT
from utils import load_configs, log_configs, send_mail

BASE_PATH = "/home/ubuntu/state2vec/"
TRACES_PATH = BASE_PATH + "data/traces"

@torch.no_grad()
def estimate_loss(model, train_dataloader, validation_dataloader, device):
    out = {}
    model.eval()
    dataloaders = {
        'train': train_dataloader,
        'val': validation_dataloader,
    }
    eval_over = len(validation_dataloader)
    for split, dataloader in dataloaders.items():
        losses = torch.zeros(eval_over)
        for k, (X, Y) in enumerate(dataloader):
            if k >= eval_over:
                break
            X, Y = X.to(device), Y.to(device)  # Ensure data is on the correct device
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def load_trace_files(path):
    logging.info("Loading trace files...")
    with open(path, "r") as EXP_FILE:
        EXP_TRACES = []
        for trace in EXP_FILE.readlines():
            trace = trace[
                    trace.find("doc/") + 4:
                    trace.find("/", trace.find("doc/") + 4)
                    ] + ".tar.gz"
            EXP_TRACES.append(trace)
    return EXP_TRACES


def main(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model_name = f"s2v-gpt_{configs.GLOBAL.exp_num}_latest.pt"

    DETAILS_JSON_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/details.json"
    DETAILS_JSON_FILE = open(DETAILS_JSON_PATH, "r")
    DETAILS_JSON = json.load(DETAILS_JSON_FILE)
    DETAILS_JSON_FILE.close()

    if os.path.exists(f'./models/{model_name}'):
        logging.info("Loading Model...")
        model = torch.load(f'./models/{model_name}')
    else:
        logging.info("Creating Model...")
        model = GPT(configs.MODEL)
        torch.save(model, f'./models/s2v-gpt_{configs.GLOBAL.exp_num}_base.pt')

    model.to(device)
    logging.info(f"# Parameters: {model.get_num_params() / 1e6:.2f}M")

    TRAINING_SET_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/training.set"
    EXP_TRACES = load_trace_files(TRAINING_SET_PATH)

    for index, trace in enumerate(EXP_TRACES):
        index += 1
        if DETAILS_JSON[trace]["trained"]:
            logging.info(f"Train History: {DETAILS_JSON[trace]["train_history"]}")
            logging.info(f"Model trained with {trace}, skipping...")
            continue

        optimizer = Adam(model.parameters(), lr=configs.TRAINING.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=configs.TRAINING.milestones,
            gamma=configs.TRAINING.gamma,
            verbose=True
        )

        logging.info(f"Trace {index}/{len(EXP_TRACES)}")
        logging.info(f"Starting for {trace}...")
        trace_file_path = TRACES_PATH + "/" + trace
        logging.info("Loading Data...")
        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=configs.MODEL.block_size
        )

        # Create DataLoaders for training
        train_dataloader = DataLoader(
            dataset,
            batch_size=configs.TRAINING.batch_size,
            shuffle=True,
            num_workers=4,  # Adjust based on your system's specification
            pin_memory=True,  # If using a GPU, this can improve transfer speeds,
            pin_memory_device="cuda",
        )

        logging.info("Starting Training...")
        history = []
        for iter in range(configs.TRAINING.epochs):
            model.train()
            with (tqdm(total=len(train_dataloader), desc=f"Epoch {iter}:") as pbar):
                for batch_idx, (Xb, Yb) in enumerate(train_dataloader):
                    Xb, Yb = Xb.to(device), Yb.to(device)
                    logits, loss = model(Xb, Yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()

                    pbar.update(1)
                    pbar.set_postfix(loss=train_loss)

            scheduler.step()
            logging.info(f"Learning Rate: {scheduler.get_last_lr()}")
            history.append(train_loss)
            logging.info(f'\tFinal Train Loss: {train_loss:.4f}')

        DETAILS_JSON[trace]["train_history"] = history

        logging.info("Training Finished! Saving model...")
        torch.save(model, f'./models/{model_name}')

        DETAILS_JSON[trace]["trained"] = True
        with open(f"../experiments/exp{configs.GLOBAL.exp_num}/details.json", "w") as f:
            json.dump(DETAILS_JSON, f)

        if index % 15 == 0 or index == 1 or index == len(EXP_TRACES):
            send_mail(f"Training continues {index}/{len(EXP_TRACES)}")

    send_mail(f"Model training done!")
    with open("../experiments/exp1/details.json", "w") as f:
        json.dump(DETAILS_JSON, f)


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