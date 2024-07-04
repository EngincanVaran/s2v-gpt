import json

import numpy as np
from loguru import logger

from utils import run_length_decode


def read_encoded_data(trace_path):
    with open(trace_path, "r") as f:
        data = f.readline().strip()
        logger.info(f"Len Trace: {len(data)}")
    data = run_length_decode(data)
    return data
    # return np.array([1 if char == 'F' else 0 for char in data])


def real_time_fraud_detection(sequence, window_size=100, threshold=0.2):
    alerts = []

    # Initialize the first window
    current_window = sequence[:window_size]
    current_sum = np.sum(current_window)  # sum is more efficient than mean for binary data
    current_avg = current_sum / window_size

    # Check the initial window
    if current_avg > threshold:
        alerts.append((0, window_size - 1))
        logger.info(f"Alert: Potential fraud detected in window 0 to {window_size - 1} with average {current_avg:.2f}")
        return "malicious"

    # Slide the window across the sequence
    for i in range(window_size, len(sequence)):
        # Update the window: remove the first element and add the new element
        current_sum = current_sum - sequence[i - window_size] + sequence[i]
        current_avg = current_sum / window_size

        # Check the new window
        if current_avg > threshold:
            alerts.append((i - window_size + 1, i))
            logger.info(f"Alert: Potential fraud detected in window {i - window_size + 1} to {i} with average {current_avg:.2f}")
            return "malicious"

    return "benign"

class CUSUM:
    def __init__(self, target, threshold, drift):
        self.target = target
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0
        self.cusum_neg = 0

    def update(self, value):
        self.cusum_pos = max(0, self.cusum_pos + (value - self.target - self.drift))
        self.cusum_neg = min(0, self.cusum_neg + (value - self.target + self.drift))

        if self.cusum_pos > self.threshold or self.cusum_neg < -self.threshold:
            return True  # Anomaly detected
        return False  # No anomaly


def process_stream(data_stream, window_size, cusum_detector):
    for i in range(0, len(data_stream) - window_size + 1, window_size):
        chunk = data_stream[i:i + window_size]
        f_count = chunk.count('F') / window_size  # Normalize count to get the rate

        # Update CUSUM and check for anomalies
        if cusum_detector.update(f_count):
            print(f"Anomaly detected at window starting at index {i}, F count rate: {f_count}")


def main():
    with open("benign_malicious_info.json", "r") as f:
        benign_malicious_info = json.load(f)

    trace = "0b2db97f949f175cf873ae8d8e276e672b13df77a7d5e5c840c62f2523fe2bcb"
    # trace = "0ba7af3a40aad4199f093e204fff17028a358a0dbe3f5d6491bb2f9270576d4a"
    trace_path = f"results/exp1_top42/{trace}.tar.gz.prediction_string"
    logger.info(f"Truth: {benign_malicious_info[trace]}")
    prediction_string = read_encoded_data(trace_path)

    # Initialize CUSUM parameters
    target = 0.1  # Example target value, expected rate of 'F' values
    threshold = 5  # Example threshold for raising alarms
    drift = 0.5  # Example drift
    # Initialize CUSUM detector
    cusum_detector = CUSUM(target, threshold, drift)
    process_stream(prediction_string, 5000, cusum_detector)

    # label = real_time_fraud_detection(prediction_string, window_size=10000, threshold=0.9)

    # logger.info(f"Label: {label}")


if __name__ == "__main__":
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    logger.info("***** Running Evaluation *****")
    main()
    logger.info("***** End Evaluation *****")