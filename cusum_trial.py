import numpy as np
import matplotlib.pyplot as plt


def compute_cusum(sequence, target_mean, slack_value):
    cusum_positive = np.zeros(len(sequence))
    cusum_negative = np.zeros(len(sequence))

    for i in range(1, len(sequence)):
        cusum_positive[i] = max(0, cusum_positive[i - 1] + (sequence[i] - target_mean - slack_value))
        cusum_negative[i] = min(0, cusum_negative[i - 1] + (sequence[i] - target_mean + slack_value))

    return cusum_positive, cusum_negative


def detect_fraud_cusum(sequence, target_mean=0.1, slack_value=0.05, decision_interval=5):
    sequence_binary = np.array([1 if char == 'F' else 0 for char in sequence])
    cusum_positive, cusum_negative = compute_cusum(sequence_binary, target_mean, slack_value)

    fraud_indices_positive = np.where(cusum_positive > decision_interval)[0]
    fraud_indices_negative = np.where(cusum_negative < -decision_interval)[0]

    return fraud_indices_positive, fraud_indices_negative, cusum_positive, cusum_negative


def flag_malicious_segments(fraud_indices, segment_length=10):
    flagged_segments = []
    for index in fraud_indices:
        start = max(0, index - segment_length // 2)
        end = min(len(long_string), index + segment_length // 2)
        flagged_segments.append((start, end))
    return flagged_segments


# Example usage
long_string = "TTFFTFTFTFTTFTFTFFFFFTTTTFTFTFTFTFTFTF" * 100  # Extend the string for a longer example
target_mean = 0.4
slack_value = 0.2
decision_interval = 5

fraud_indices_positive, fraud_indices_negative, cusum_positive, cusum_negative = detect_fraud_cusum(
    long_string, target_mean, slack_value, decision_interval
)

print("Potential fraud detected at positions (positive CUSUM):", fraud_indices_positive)
print("Potential fraud detected at positions (negative CUSUM):", fraud_indices_negative)

flagged_segments_positive = flag_malicious_segments(fraud_indices_positive)
flagged_segments_negative = flag_malicious_segments(fraud_indices_negative)

print("Flagged malicious segments (positive CUSUM):", flagged_segments_positive)
print("Flagged malicious segments (negative CUSUM):", flagged_segments_negative)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(cusum_positive, label='CUSUM Positive', color='blue')
plt.plot(cusum_negative, label='CUSUM Negative', color='red')
plt.axhline(y=decision_interval, color='green', linestyle='--', label='Decision Interval')
plt.axhline(y=-decision_interval, color='green', linestyle='--')
plt.xlabel('Position in Sequence')
plt.ylabel('CUSUM Value')
plt.title('CUSUM Control Chart for Fraud Detection')
plt.legend()
plt.savefig("dummy.png")
