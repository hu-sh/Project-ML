import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Read the file
# Skipping comment lines starting with '#' and assuming no header row
file_path = 'data/CUP/ML-CUP25-TR.csv'
df = pd.read_csv(file_path, comment='#', header=None)

# 2. Extract inputs and targets based on the column indices
# Column 0: ID
# Columns 1-12: Inputs (x1 is index 1, x2 is index 2)
# Columns 13-16: Targets (y3 is index 15, y4 is index 16)
x1 = df[1]
x2 = df[6]
y3 = df[15]
y4 = df[16]

# 3. Calculate z
z = y3 - y4

# 4. Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using z for the color mapping
scatter = ax.scatter(x1, x2, z, c=z, cmap='viridis', marker='o', alpha=0.7)

# Add labels and title
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$z = y_3 - y_4$')
ax.set_title('3D Plot of $x_1, x_2, z$')

# Optional: add a color bar
fig.colorbar(scatter, ax=ax, label='$z$ value')

plt.tight_layout()
plt.show()
