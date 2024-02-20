import logging

import torch

from hex_dataset import HexDataset
from model import GPT
from utils import encode, decode, instantiate_configs

BASE_PATH = "/home/ubuntu/state2vec/"
TRACES_PATH = BASE_PATH + "data/traces"


def main():
    logging.info(f"Cuda Available: {torch.cuda.is_available()}")

    model = torch.load("./models/s2v-gpt_latest.pth")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    TEST_SET_PATH = f"experiments/exp1/test.set"
    EXP_FILE = open(TEST_SET_PATH)
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

        logging.info(f"Trace {index}/{len(EXP_TRACES)}")
        logging.info(f"Starting for {trace}")
        trace_file_path = TRACES_PATH + "/" + trace

        dataset = HexDataset(
            file_path=trace_file_path,
            block_size=32
        )

        for x, y in dataset:
            print(x, y)
            exit()


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
