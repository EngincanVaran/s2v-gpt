import json
import os

from loguru import logger
from tqdm import tqdm

from utils import run_length_decode, send_mail


class CUSUM:
    def __init__(self, target, threshold, drift):
        self.target = target
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0
        # self.cusum_neg = 0

    def update(self, value):
        self.cusum_pos = max(0, self.cusum_pos + (value - self.target - self.drift))
        # self.cusum_neg = min(0, self.cusum_neg + (value - self.target + self.drift))

        if self.cusum_pos > self.threshold: # or self.cusum_neg < -self.threshold:
            return True  # Anomaly detected
        return False  # No anomaly


def process_stream(data_stream, window_size, cusum_detector):
    for i in range(0, len(data_stream) - window_size + 1, window_size):
        chunk = data_stream[i:i + window_size]
        f_count = chunk.count('F') / window_size  # Normalize count to get the rate

        # Update CUSUM and check for anomalies
        if cusum_detector.update(f_count):
            # logger.info(f"Anomaly detected at window starting at index {i}, F count rate: {f_count}")
            return "malicious"
    return "benign"


def read_encoded_data(trace_path):
    with open(trace_path, "r") as f:
        data = f.readline().strip()
    return run_length_decode(data)


def main():
    # target = 0.3  # Example target value, expected rate of 'F' values
    # threshold = 5  # Example threshold for raising alarms
    # drift = 0.2  # Example drift
    window_size = 50000

    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)

    TRACES = os.listdir("results/exp1_top42")

    target_values = [0.15, 0.2, 0.25]
    threshold_values = [3, 4, 5]
    drift_values = [0.2]
    for target in target_values:
        for threshold in threshold_values:
            for drift in drift_values:
                logger.info(f"Configs:\n"
                            f"\t\t\t\tWindowSize: {window_size}"
                            f"\tTarget: {target}"
                            f"\tThreshold: {threshold}"
                            f"\tDrift: {drift}")

                results = {}
                truth_string = ""
                pred_string = ""
                with tqdm(total=len(TRACES), desc=f"Evaluating...") as pbar:
                    for idx, trace in enumerate(TRACES):
                        index = trace.find(".tar.gz")
                        trace_name = trace[:index]
                        trace_path = f"results/exp1_top42/{trace}"
                        prediction_string = read_encoded_data(trace_path)

                        # Initialize CUSUM detector
                        cusum_detector = CUSUM(target, threshold, drift)
                        label = process_stream(prediction_string, window_size, cusum_detector)
                        truth = benign_malicious_info[trace_name]
                        results[trace_name] = {
                            "truth": truth,
                            "label": label
                        }
                        truth_string += "|" if truth == "benign" else ":"
                        pred_string += "|" if label == "benign" else ":"

                        pbar.update(1)

                        # if idx % 200 == 0 or idx == len(TRACES):
                        #    send_mail(f"Evaluation continues {idx}/{len(TRACES)}")

                    # send_mail(f"Evaluations done!")
                    with open(f"eval_results/exp1_cusum/results_"
                              f"{window_size}_"
                              f"{target}_"
                              f"{threshold}_"
                              f"{drift}.json", "w") as f:
                        json.dump(results, f)
                    logger.info(truth_string)
                    logger.info(pred_string)
    send_mail("Evaluations done!")


if __name__ == "__main__":
    logger.info("***** Running Evaluation CUSUM *****")
    main()
    logger.info("***** End Evaluation CUSUM *****")

'''
TN FP
FN TN

Reducing False Negatives:
- Decrease Threshold: Increases sensitivity to detect more fraud.
- Decrease Drift: More responsive to small deviations.
- Decrease Target: Lower baseline makes smaller deviations detectable.
- Decrease Window Size: Captures short-term anomalies.
Reducing False Positives:
- Increase Threshold: Decreases sensitivity to avoid false alarms.
- Increase Drift: Tolerates minor fluctuations without flagging them.
- Increase Target: Higher baseline requires larger deviations for detection.
- Increase Window Size: Smooths out noise by considering larger data segments.
'''