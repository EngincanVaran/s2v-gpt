from hex_dataset import HexDataset

FILE_PATH = "../benign.066165f874547a1cfabce372f202b70bc49f048e1d9a3b758b81df8fa549bd70.trace_12bit.txt"

dataset = HexDataset(
    file_path=FILE_PATH,
    block_size=32
)

for x, y in dataset:
    print(x, y)
    exit()