import json
import logging
import os
import time

from tqdm import tqdm

from utils import load_configs, log_configs, run_length_decode, send_mail


def apply_sliding_window(data, suspiciousWindowSize, suspicious_threshold):
    window_labels = []

    for i in range(0, len(data), suspiciousWindowSize):
        window = data[i:i + suspiciousWindowSize]
        if len(window) < suspiciousWindowSize:
            break  # Stop if the window is smaller than the expected size

        s = window.count("F")
        label = "0"
        if s > (suspiciousWindowSize * suspicious_threshold):
            label = "1"

        window_labels.append(label)

    return window_labels


def main(configs):
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)

    TRACES = os.listdir("results/exp1_top42")
    DONE_TRACES = os.listdir("second_predictions/exp1")

    with (tqdm(total=len(TRACES), desc=f"Evaluating...") as pbar):
        for index, trace in enumerate(TRACES):
            index = trace.find(".tar.gz")
            trace_name = trace[:index]

            if f"second_predictions/exp1_top42/{trace_name}.second_prediction_string" in DONE_TRACES:
                pbar.update(1)
                continue
            with open(f"results/exp1_top42/{trace}") as f:
                prediction_string = f.readline().strip()

            prediction_string = run_length_decode(prediction_string)

            TRUTH = benign_malicious_info[trace_name]

            labelled_windows = apply_sliding_window(
                prediction_string,
                configs.EVALUATION.window_size,
                configs.EVALUATION.suspicious_threshold
            )

            with open(f"second_predictions/exp1/{trace_name}.second_prediction_string", "w") as f:
                f.write("".join(labelled_windows))

            pbar.update(1)

            if index % 15 == 0 or index == 1 or index == len(TRACES):
                send_mail(f"Prediction continues {index}/{len(TRACES)}")

    send_mail(f"Second Prediction Data Done!")



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
    logging.info("***** Running Evaluation *****")

    configs = load_configs("configs.yaml")
    log_configs(configs)

    main(configs)

    logging.info("***** End Evaluation *****")
