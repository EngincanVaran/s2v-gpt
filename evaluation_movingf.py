import json

import os

from loguru import logger
from tqdm import tqdm

from utils import run_length_decode

import numpy as np


def real_time_fraud_detection(sequence, window_size=100, threshold=0.2):
    # Initialize the first window
    current_window = sequence[:window_size]
    current_sum = np.sum(current_window)  # sum is more efficient than mean for binary data
    current_avg = current_sum / window_size

    # Check the initial window
    if current_avg > threshold:
        # logger.info(f"Alert: Potential fraud detected in window 0 to {window_size - 1} with average {current_avg:.2f}")
        return "malicious"

    # Slide the window across the sequence
    for i in range(window_size, len(sequence)):
        # Update the window: remove the first element and add the new element
        current_sum = current_sum - sequence[i - window_size] + sequence[i]
        current_avg = current_sum / window_size

        # Check the new window
        if current_avg > threshold:
            # logger.info(f"Alert: Potential fraud detected in window {i - window_size + 1} to {i} with average {current_avg:.2f}")
            return "malicious"

    return "benign"


def read_encoded_data(trace_path):
    with open(trace_path, "r") as f:
        data = f.readline().strip()
    data = run_length_decode(data)
    data = np.array([1 if char == 'F' else 0 for char in data])
    return data


def main():
    window_size = 5000
    suspicious_threshold = 0.8
    lag = 5000
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)

    results = {}
    truth_string = ""
    pred_string = ""

    TRACES = os.listdir("results/exp1_top42")
    with tqdm(total=len(TRACES), desc=f"Evaluating...") as pbar:
        for idx, trace in enumerate(TRACES):
            index = trace.find(".tar.gz")
            trace_name = trace[:index]
            truth = benign_malicious_info[trace_name]
            trace_path = f"results/exp1_top42/{trace}"

            prediction_string = read_encoded_data(trace_path)

            label = real_time_fraud_detection(prediction_string, window_size=1000, threshold=0.8)
            results[trace_name] = {
                "truth": truth,
                "label": label
            }
            truth_string += "|" if truth == "benign" else ":"
            pred_string += "|" if label == "benign" else ":"

            pbar.update(1)

            if idx % 100 == 0 or idx == len(TRACES):
                # send_mail(f"Evaluation continues {idx}/{len(TRACES)}")
                continue

        # send_mail(f"Evaluations done!")
        with open(f"eval_results/exp1/movingf_results_"
                  f"{window_size}_"
                  f"{suspicious_threshold}_"
                  f"{lag}.json", "w") as f:
            json.dump(results, f)
        logger.info(truth_string)
        logger.info(pred_string)


if __name__ == "__main__":
    logger.info("***** Running Evaluation *****")
    main()
    logger.info("***** End Evaluation *****")
