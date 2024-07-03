import json
import logging

import torch
from tqdm import tqdm

from hex_dataset import HexDataset
from torch.utils.data import DataLoader

from utils import load_configs, log_configs, send_mail, run_length_encode

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
    model_name = f"s2v_gpt_{configs.GLOBAL.exp_num}_latest.pt"

    DETAILS_JSON_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/pred_details.json"
    DETAILS_JSON_FILE = open(DETAILS_JSON_PATH, "r")
    DETAILS_JSON = json.load(DETAILS_JSON_FILE)
    DETAILS_JSON_FILE.close()

    logging.info("Loading Model...")
    model = torch.load(f'./models/{model_name}')

    model.to(device)
    logging.info(f"# Parameters: {model.get_num_params() / 1e6:.2f}M")

    TEST_SET_PATH = f"experiments/exp{configs.GLOBAL.exp_num}/test.set"
    EXP_TRACES = load_trace_files(TEST_SET_PATH)

    for index, trace in enumerate(EXP_TRACES):
        index += 1
        if DETAILS_JSON[trace]:
            logging.info(f"Model predicted with {trace}, skipping...")
            continue

        logging.info(f"Trace {index}/{len(EXP_TRACES)}")
        logging.info(f"Starting for {trace}...")
        trace_file_path = TRACES_PATH + "/" + trace
        logging.info("Loading Data...")
        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=configs.PREDICTION.block_size
        )

        prediction_dataloader = DataLoader(
            dataset,
            batch_size=configs.PREDICTION.batch_size,
            shuffle=False,
            num_workers=configs.PREDICTION.max_workers,  # Adjust based on your system's specification
            pin_memory=True,  # If using a GPU, this can improve transfer speeds
            pin_memory_device="cuda",
        )

        logging.info("Starting Predictions...")
        model.eval()
        prediction_string = ""
        with torch.no_grad():
            with tqdm(total=len(prediction_dataloader), desc=f"Predicting...") as pbar:
                for batch_idx, (Xb, y_truth) in enumerate(prediction_dataloader):
                    y_truth = y_truth[:, -1].to(device)
                    Xb = Xb.to(device)
                    # logging.info(f"XB: {Xb[0]}")
                    # logging.info(f"Len XB: {len(Xb[0])}")
                    y_pred = model.get_next_word_probs(Xb)
                    # logging.info(f"Y_PRED: {y_pred[0]}")
                    # logging.info(f"Y_TRUTH: {y_truth}")
                    values, indices = torch.topk(y_pred, k=configs.PREDICTION.top_k, dim=1)
                    # logging.info(f"Y_INDICES: {indices[0]}")
                    # Ensure y_truth has the correct shape for comparison
                    y_truth_expanded = y_truth.unsqueeze(1).expand_as(indices)
                    # logging.info(f"Y_TRUE_INDICES: {y_truth[0]}")
                    # Use broadcasting to check if the true indices are in the top-k predicted indices
                    matches = torch.any(indices == y_truth_expanded, dim=1)
                    # logging.info(f"Matches: {matches[0]}")
                    prediction_string += ''.join(['T' if match else 'F' for match in matches])
                    # logging.info(f"Prediction String: {prediction_string}")
                    pbar.update(1)

        logging.info("Writing Predictions!")
        with open(f'results/exp{configs.GLOBAL.exp_num}/{trace}.prediction_string', "w") as f:
            f.write(run_length_encode(prediction_string))

        DETAILS_JSON[trace] = True
        logging.info("Prediction Finished...")
        with open(f"experiments/exp{configs.GLOBAL.exp_num}/pred_details.json", "w") as f:
            json.dump(DETAILS_JSON, f)
        prediction_string = ""
        if index % 15 == 0 or index == 1 or index == len(EXP_TRACES):
            send_mail(f"Prediction continues {index}/{len(EXP_TRACES)}")

    send_mail(f"Model Predictions Done!")
    with open("experiments/exp1/pred_details.json", "w") as f:
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
    logging.info("***** Running Prediction *****")

    configs = load_configs("configs.yaml")
    log_configs(configs)

    logging.info(f"Cuda Available: {torch.cuda.is_available()}")

    main(configs)

    logging.info("***** End Prediction *****")