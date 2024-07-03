import json
import logging
import os
from tqdm import tqdm

from utils import load_configs, run_length_decode, send_mail


def apply_sliding_window(data, suspiciousWindowSize, lag, suspicious_threshold):
    max_s = 0
    label = "benign"

    for i in range(0, len(data) - suspiciousWindowSize, lag):
        window = data[i:i + suspiciousWindowSize]
        s = window.count("F")
        if s >= (suspiciousWindowSize * suspicious_threshold):
            label = "malicious"
            return label, max_s
        if s > max_s:
            max_s = s
    return label, max_s


def read_encoded_data(trace_path):
    with open(trace_path, "r") as f:
        data = f.readline().strip()

    return run_length_decode(data)

def main(configs):
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)

    results = {}

    TRACES = os.listdir("results/exp1_top42")
    with tqdm(total=len(TRACES), desc=f"Evaluating...") as pbar:
        for idx, trace in enumerate(TRACES):
            index = trace.find(".tar.gz")
            trace_name = trace[:index]
            trace_path = f"results/exp1_top42/{trace}"

            prediction_string = read_encoded_data(trace_path)

            label, max_s = apply_sliding_window(
                prediction_string,
                configs.EVALUATION.window_size,
                configs.EVALUATION.lag,
                configs.EVALUATION.suspicious_threshold
            )
            results[trace_name] = {
                "truth": benign_malicious_info[trace_name],
                "label": label,
                "max_s": max_s
            }

            pbar.update(1)

            if idx % 100 == 0 or idx == len(TRACES):
                send_mail(f"Evaluation continues {idx}/{len(TRACES)}")

        send_mail(f"Evaluations done!")
        with open(f"eval_results/exp1/results_"
                  f"{configs.EVALUATION.window_size}_"
                  f"{configs.EVALUATION.suspicious_threshold}_"
                  f"{configs.EVALUATION.lag}.json", "w") as f:
            json.dump(results, f)


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
    # log_configs(configs)

    main(configs)

    logging.info("***** End Evaluation *****")
