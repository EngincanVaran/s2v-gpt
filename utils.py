import smtplib
from dataclasses import fields

import numpy as np
import yaml

from gpt_configs import GPTConfig


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


def instantiate_configs(yaml_path: str) -> GPTConfig:
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Ensure only valid attributes for GPTConfig are passed
    valid_attrs = {field.name for field in fields(GPTConfig)}
    filtered_config_data = {k: v for k, v in config_data.items() if k in valid_attrs}

    return GPTConfig(**filtered_config_data)


def send_mail(body, subject="S2V-GPT Update!"):
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


def find_order_of_element(array, index):
    # Sort the array in descending order while keeping track of original indices
    sorted_indices = np.argsort(array)

    # Find the order of the element with the given index
    order = np.where(sorted_indices == index)[0][0] + 1  # Adding 1 to make it 1-based indexing

    return order
