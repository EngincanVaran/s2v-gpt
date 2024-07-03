import dataclasses
import logging
import smtplib
from dataclasses import fields

import numpy as np
import yaml

from gptconfig import TrainingConfigs, ModelConfigs, Config, GlobalConfigs, PredictionConfigs, EvaluationConfigs


def generate_hex_range():
    return [format(i, '03x').lower() for i in range(4096)]


chars = generate_hex_range()
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """
    Encoder function: Takes a string and outputs a list of integers.

    Args:
    s (str): Input string to encode.

    Returns:
    list: List of integers representing the encoded string.
    """
    return [stoi[c] for c in s]


def decode(l):
    """
    Decoder function: Takes a list of integers and outputs a string.

    Args:
    l (list): List of integers to decode.

    Returns:
    str: Decoded string.
    """
    if isinstance(l, int):
        l = [l]
    return ' '.join([itos[i] for i in l])


def load_configs(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    global_configs = GlobalConfigs(**config['global_configs'])
    training_configs = TrainingConfigs(**config['training_configs'])
    model_configs = ModelConfigs(**config['model_configs'])
    prediction_configs = PredictionConfigs(**config['prediction_configs'])
    evaluation_configs = EvaluationConfigs(**config['evaluation_configs'])
    return Config(
        global_configs,
        training_configs,
        model_configs,
        prediction_configs,
        evaluation_configs
    )


def log_configs(config):
    max_line_length = 60  # Adjust this based on your longest config line
    logging.info("+" + "-" * (max_line_length + 2) + "+")
    logging.info("|" + " Configs ".center(max_line_length + 2) + "|")
    logging.info("+" + "-" * (max_line_length + 2) + "+")

    for field in fields(config):
        field_name = field.name
        field_value = getattr(config, field_name)

        if dataclasses.is_dataclass(field_value):
            logging.info("|" + f" {field_name.upper()} ".ljust(max_line_length) + "       |")
            for sub_field in fields(field_value):
                sub_field_name = sub_field.name
                sub_field_value = getattr(field_value, sub_field_name)
                logging.info("|" + f"\t{sub_field_name.upper()}: {sub_field_value}".ljust(max_line_length) + "|")
        else:
            logging.info("|" + f" {field_name.upper()}: {field_value}".ljust(max_line_length) + "|")

    logging.info("+" + "-" * (max_line_length + 2) + "+")


def send_mail(body, subject="A100 S2V-GPT Update!"):
    BOT_PASSWORD = "govfwswxpxwkoziq"
    BOT_EMAIL = "crypto.bot.penguin@gmail.com"
    EMAIL = "evaran@sabanciuniv.edu"

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        smtp.login(BOT_EMAIL, BOT_PASSWORD)

        msg = f'Subject: {subject} \n\n{body}'

        smtp.sendmail(BOT_EMAIL, EMAIL, msg)


def run_length_encode(data):
    encoding = []
    prev_char = data[0]
    count = 1

    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding.append(f"{prev_char}{count}")
            prev_char = char
            count = 1
    encoding.append(f"{prev_char}{count}")

    return ''.join(encoding)


def run_length_decode(encoded_data):
    decoded = []
    i = 0
    while i < len(encoded_data):
        char = encoded_data[i]
        count = ''
        i += 1
        while i < len(encoded_data) and encoded_data[i].isdigit():
            count += encoded_data[i]
            i += 1
        decoded.append(char * int(count))
    return ''.join(decoded)
