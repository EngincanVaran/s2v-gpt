import logging
import os.path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_trace(trace, device, model, configs):
    logging.info(f"Starting for {trace}...")
    trace_file_path = TRACES_PATH + "/" + trace
    trace_dataset = HexDataset(trace_file_path)
    trace_dataloader = DataLoader(trace_dataset, batch_size=1, shuffle=False)

    prediction_string = ""
    for batch_idx, (Xb, Yb) in enumerate(tqdm(trace_dataloader, desc="Processing batches")):
        Xb, Yb = Xb.to(device), Yb.to(device)
        y_pred = model.get_next_word_probs(Xb)
        values, indices = torch.topk(y_pred, k=configs.PREDICTION.top_k, dim=1)
        matches = torch.any(indices == Yb[:, 0].unsqueeze(1), dim=1)
        prediction_string += ''.join(['T' if match else 'F' for match in matches])

    logging.info("Writing Predictions!")
    with open(f'results/exp{configs.GLOBAL.exp_num}/{trace}.prediction_string', "w") as f:
        f.write(prediction_string)

    # Assuming 'details.json' keeps track of which traces have been processed
    with open(f"experiments/exp{configs.GLOBAL.exp_num}/details.json", "w") as f:
        details_json = json.load(f)
        details_json[trace]["predicted"] = True
        json.dump(details_json, f)


def main(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "s2v-gpt_1_latest.pt"
    model = torch.load(f'./models/{model_name}')
    model.to(device)

    EXP_TRACES = load_trace_files(f"experiments/exp{configs.GLOBAL.exp_num}/test.set")

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_trace = {executor.submit(process_trace, trace, device, model, configs): trace for trace in EXP_TRACES}

        for future in as_completed(future_to_trace):
            trace = future_to_trace[future]
            try:
                future.result()  # to catch exceptions if any
            except Exception as exc:
                logging.error(f'{trace} generated an exception: {exc}')
            else:
                logging.info(f'{trace} processed successfully.')


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
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
