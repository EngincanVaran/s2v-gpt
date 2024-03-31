import logging

import torch

from hex_dataset import HexDataset
import heapq
from utils import decode

BASE_PATH = "/home/ubuntu/state2vec/"
TRACES_PATH = BASE_PATH + "data/traces"


def main():
    logging.info(f"Cuda Available: {torch.cuda.is_available()}")

    model = torch.load("s2v-gpt.pth", map_location=torch.device('cpu'))
    logging.info("Model Loaded - s2v-gpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trace_file_path = "../387995ba4dd17a50eba3b51455ad6d52329e940dc7cfca23c3b5e766d93a5148.tar.gz"

    dataset = HexDataset(
        file_path=trace_file_path,
        block_size=32
    )

    for x, y in dataset:
        xb = x.to(device)
        predictions = model.next_word_prob(xb)[0]
        top_20_indices = heapq.nlargest(20, range(len(predictions)), key=predictions.__getitem__)

        # Decode these indices
        decoded_indices = decode(top_20_indices)
        print("Top 20 Indices (Decoded):", decoded_indices)
        print(y[0])
        print(decode(y[0].tolist()))
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
