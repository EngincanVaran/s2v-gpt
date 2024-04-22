import logging
import json
import os


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


def main():
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)
    results = {}
    trace_list = os.listdir("results/exp1")

    for index, trace in enumerate(trace_list):
        trace_path = "results/exp1/" + trace
        index = trace.find(".tar.gz")
        trace_name = trace[:index]
        TRUTH = benign_malicious_info[trace_name]
        logging.info(f"Starting for {trace_name} | {TRUTH}")
        data = read_trace(trace_path)
        label, max_s = apply_sliding_window(data, 50000, 1, 0.9)
        logging.info(f"Label: {label}\tMax_Suspicious: {max_s}")
        results[trace_name] = {
            "Truth": TRUTH,
            "Label": label,
            "MaxSuspicious": max_s
        }

    with open("results/exp1/results.json", "w") as f:
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
        logging.info("***** Running Training *****")
        main()

        logging.info("***** End Training *****")