import logging
import os.path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hex_dataset import HexDataset
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

    DETAILS_JSON_PATH = f"results/exp{configs.GLOBAL.exp_num}/prediction_details.json"
    DETAILS_JSON_FILE = open(DETAILS_JSON_PATH, "r")
    DETAILS_JSON = json.load(DETAILS_JSON_FILE)
    DETAILS_JSON_FILE.close()

    if os.path.exists(f'./models/{model_name}'):
        logging.info("Loading Model...")
        model = torch.load(f'./models/{model_name}')
    else:
        logging.error("Model Not Found!")
        return

    model.to(device)
    logging.info(f"# Parameters: {model.get_num_params() / 1e6:.2f}M")

    TESTING_SET_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/test.set"
    EXP_TRACES = load_trace_files(TESTING_SET_PATH)

    for index, trace in enumerate(EXP_TRACES):
        index += 1
        if DETAILS_JSON[trace]["predicted"]:
            logging.info(f"Model already made predictions for {trace}, skipping...")
            continue

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
            batch_size=configs.PREDICTION.batch_size,
            shuffle=False,
            num_workers=4,  # Adjust based on your system's specification
            pin_memory=True,  # If using a GPU, this can improve transfer speeds,
            pin_memory_device="cuda",
        )
        prediction_string = ""
        # Wrap your dataloader with tqdm for a progress bar
        for batch_idx, (Xb, Yb) in enumerate(tqdm(train_dataloader, desc="Processing batches:")):
            Xb, Yb = Xb.to(device), Yb.to(device)
            y_pred = model.get_next_word_probs(Xb)
            values, indices = torch.topk(y_pred, k=configs.PREDICTION.top_k, dim=1)
            # Ensure Yb is correctly reshaped for comparison
            true_indices = Yb[:, 0].unsqueeze(1)  # Adjust as necessary based on your data structure

            # Use broadcasting to check if the true indices are in the top-k predicted indices
            matches = torch.any(indices == true_indices, dim=1)
            prediction_string += ''.join(['T' if match else 'F' for match in matches])

        logging.info("Writing Predictions!")
        with open(f'results/exp{configs.GLOBAL.exp_num}/{trace}.prediction_string', "w") as f:
            f.write(prediction_string)

        DETAILS_JSON[trace]["predicted"] = True
        with open(f"experiments/exp{configs.GLOBAL.exp_num}/details.json", "w") as f:
            json.dump(DETAILS_JSON, f)

        if index % 15 == 0 or index == 1 or index == len(EXP_TRACES):
            send_mail(f"Prediction continues {index}/{len(EXP_TRACES)}")

    send_mail(f"Predictions done!")
    with open(f"../results/exp{configs.GLOBAL.exp_num}/prediction_details.json", "w") as f:
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