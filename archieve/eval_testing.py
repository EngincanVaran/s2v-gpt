import numpy as np
import torch
import heapq

# Correct way to initialize predictions as a numpy array
from utils import decode


print(decode([2330]))
exit()

predictions = np.random.rand(4096)

# Convert the numpy array to a PyTorch tensor (if needed for other purposes)
predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

# Find the indices of the top 20 maximum values in predictions
top_20_indices = heapq.nlargest(20, range(len(predictions)), key=predictions.__getitem__)

# Decode these indices
decoded_indices = decode(top_20_indices)

print("Top 20 Indices (Decoded):", decoded_indices)
