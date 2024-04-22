import logging
import json
import os
from tqdm import tqdm
from utils import load_configs, log_configs, send_mail


def read_trace(path):
    data = []
    with open(path, "r") as f:
        for temp in f.readlines():
            for t in temp:
                data.append(t)
    return data

def apply_sliding_window(data, suspiciousWindowSize, lag, suspicious_threshold):
    max_s = 0
    label = "benign"

    for i in range(0, len(data) - suspiciousWindowSize, lag):
        window = data[i:i + suspiciousWindowSize]
        s = window.count("F")
        if s > (suspiciousWindowSize * suspicious_threshold):
            label = "malicious"
            return label, s
        if s > max_s:
            max_s = s
    return label, max_s


def main(configs):
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)
    results = {}

    trace_list = [trace for trace in os.listdir("results/exp1") if ".tar.gz" in trace]

    with (tqdm(total=len(trace_list), desc="Evaluating") as pbar):
        for idx, trace in enumerate(trace_list):
            trace_path = "results/exp1/" + trace
            index = trace.find(".tar.gz")
            trace_name = trace[:index]
            TRUTH = benign_malicious_info[trace_name]

            pbar.set_postfix(trace=trace_name)

            data = read_trace(trace_path)
            label, max_s = apply_sliding_window(
                data,
                configs.EVALUATION.window_size,
                configs.EVALUATION.lag,
                configs.EVALUATION.suspicious_threshold
            )

            # logging.info(f"Label: {label}\tMax_Suspicious: {max_s}")
            results[trace_name] = {
                "Truth": TRUTH,
                "Label": label,
                "MaxSuspicious": max_s
            }

            pbar.update(1)
            if idx % 15 == 0 or idx == 1 or idx == len(trace_list):
                send_mail(f"Evaluating Continues {idx}/{len(trace_list)}")

    with open(f"eval_results/exp1/eval_results_ws{configs.EVALUATION.window_size}_st{configs.EVALUATION.suspicious_threshold}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
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