import json
import logging
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, precision_score, confusion_matrix
import pandas as pd


def main():
    with open(f"eval_results/exp1_cusum/results_50000_0.2_5_0.2.json", "r") as f:
        results = json.load(f)

    truths = []
    labels = []

    for key, value in results.items():
        truths.append(value["truth"])
        labels.append(value["label"])

    logging.info(f"Benign Counts: {labels.count("benign")}")
    logging.info(f"Malicious Counts: {labels.count("malicious")}")

    logging.info(f"Accuracy Score: {accuracy_score(truths, labels)}")
    logging.info(f"Balanced Accuracy Score: {balanced_accuracy_score(truths, labels)}")

    logging.info(f"Benign F1 Score: {f1_score(truths, labels, pos_label="benign", zero_division=0)}")
    logging.info(f"Benign Precision Score: {precision_score(truths, labels, pos_label="benign", zero_division=0)}")
    logging.info(f"Benign Recall Score: {recall_score(truths, labels, pos_label="benign", zero_division=0)}")

    logging.info(f"Malicious F1 Score: {f1_score(truths, labels, pos_label="malicious", zero_division=0)}")
    logging.info(f"Malicious Precision Score: {precision_score(truths, labels, pos_label="malicious", zero_division=0)}")
    logging.info(f"Malicious Recall Score: {recall_score(truths, labels, pos_label="malicious", zero_division=0)}")

    # Compute the confusion matrix
    cm = confusion_matrix(truths, labels, labels=['benign', 'malicious'])

    # Create a DataFrame for the confusion matrix
    df_cm = pd.DataFrame(cm, index=['Actual Benign', 'Actual Malicious'],
                         columns=['Predicted Benign', 'Predicted Malicious'])

    # Log the confusion matrix
    logging.info(f"\nConfusion Matrix:\n{df_cm}")


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

    main()