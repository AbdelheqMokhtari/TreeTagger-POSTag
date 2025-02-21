import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the CSV file name
csv_file = "train_model_confusion_matrix_2.csv"

# Read the CSV file (ensure the file is in the correct directory)
df = pd.read_csv(csv_file, index_col=0)

# Create a colormap using the specified color as the base.
cmap = sns.light_palette("#DD8452", as_cmap=True)

# Set up the plot size
plt.figure(figsize=(10, 8))

# Generate the heatmap with annotations for each cell.
sns.heatmap(df, annot=True, fmt="d", cmap=cmap, cbar=True)

# Add title and axis labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Save the figure with the same base name as the CSV file (with .png extension)
png_file = os.path.splitext(csv_file)[0] + ".png"
plt.savefig(png_file, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
