import numpy as np


def real_time_fraud_detection(sequence, window_size=100, threshold=0.2):
    sequence_binary = np.array([1 if char == 'F' else 0 for char in sequence])
    alerts = []

    # Initialize the first window
    current_window = sequence_binary[:window_size]
    current_avg = np.mean(current_window)

    # Check the initial window
    if current_avg > threshold:
        alerts.append((0, window_size - 1))
        print(f"Alert: Potential fraud detected in window 0 to {window_size - 1} with average {current_avg:.2f}")

    # Slide the window across the sequence
    for i in range(window_size, len(sequence_binary)):
        # Update the window: remove the first element and add the new element
        current_window = np.append(current_window[1:], sequence_binary[i])
        current_avg = np.mean(current_window)

        # Check the new window
        if current_avg > threshold:
            alerts.append((i - window_size + 1, i))
            print(
                f"Alert: Potential fraud detected in window {i - window_size + 1} to {i} with average {current_avg:.2f}")

    return alerts


# Example usage
long_string = "TTFFTFTFTFTTFTFTFFFFFTTTTFTFTFTFTFTFTF" * 100  # Extend the string for a longer example
window_size = 100
threshold = 0.2

alerts = real_time_fraud_detection(long_string, window_size, threshold)
print("Alerts detected at windows:", alerts)
