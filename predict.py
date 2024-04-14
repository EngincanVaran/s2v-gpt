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
    model_name = f"s2v-gpt_1_latest.pt"

    # if os.path.exists(f'./models/{model_name}'):
    if os.path.exists(f'{model_name}'):
        logging.info("Loading Model...")
        # model = torch.load(f'./models/{model_name}')
        model = torch.load(f'{model_name}', map_location=torch.device('cpu'))
    else:
        logging.error("Model Not Found!")
        return

    model.to(device)
    logging.info(f"# Parameters: {model.get_num_params() / 1e6:.2f}M")

    random_tensor = torch.randint(0, configs.MODEL.vocab_size, (configs.TRAINING.batch_size, configs.MODEL.block_size))

    y_pred = model.get_next_word_probs(random_tensor)
    values, indices = torch.topk(y_pred, 3, dim=1)
    print(values, indices)


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