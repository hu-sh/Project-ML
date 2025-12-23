import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/CUP/ML-CUP25-TR.csv', skiprows=7, header=None)

# Extract targets y1 (col 13) and y2 (col 14)
y1 = df[13]
y2 = df[14]

# Extract inputs x3 (col 3) and x4 (col 4)
x1 = df[1]
x2 = df[2]

# Calculate the normalization factor: sqrt(|x3 * x4|)

# Avoid division by zero
u = y1/np.hypot(y1,y2) 
v = x1/np.hypot(x1, x2)
# Create the plot
plt.figure(figsize=(8, 8))
plt.scatter(u, v, alpha=0.5, s=15, color='darkblue')
plt.axhline(0, color='black', lw=1, alpha=0.3)
plt.axvline(0, color='black', lw=1, alpha=0.3)
plt.xlabel('y1 / sqrt(|x3 * x4|)')
plt.ylabel('y2 / sqrt(|x3 * x4|)')
plt.title('Normalized Targets: (y1, y2) / sqrt(|x3 * x4|)')
plt.grid(True, linestyle=':', alpha=0.6)

# Set symmetric axes to see the shape clearly
limit = max(np.abs(u).max(), np.abs(v).max()) * 0.5 # Zooming in a bit as outliers can stretch the view
plt.xlim(-limit, limit)
plt.ylim(-limit, limit)

plt.show()

# Calculate some stats for comparison
print(f"Max ratio for sqrt(x3*x4): {np.sqrt(u**2 + v**2).max():.4f}")
print(f"Mean magnitude after normalization: {np.sqrt(u**2 + v**2).mean():.4f}")
