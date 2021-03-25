import requests

import numpy as np
from tqdm.notebook import tqdm

def download(source_url, target_filename, chunk_size=1024):
    response = requests.get(source_url, stream=True)
    file_size = int(response.headers['Content-Length'])

    with open(target_filename, 'wb') as handle:
        for data in tqdm(response.iter_content(chunk_size=chunk_size),
                         total=int(file_size / chunk_size), unit='KB',
                         desc='Downloading dataset:'):
            handle.write(data)

def load_pendigits_dataset(filename):
    with open(filename, 'r') as f:
        data_lines = f.readlines()

    data = []
    data_labels = []
    current_digit = None

    for line in data_lines:
        if line == "\n":
            continue

        if line[0] == ".":
            if "SEGMENT DIGIT" in line[1:]:
                if current_digit is not None:
                    data.append(np.array(current_digit))
                    data_labels.append(digit_label)

                current_digit = []
                digit_label = int(line.split('"')[1])
            else:
                continue

        else:
            x, y = map(float, line.split())
            current_digit.append([x, y])
            
    data.append(np.array(current_digit))
    data_labels.append(digit_label)

    return data, data_labels
