import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Define folder paths
data_folder = "Data"
output_folder = "statistical_results"
output_file = os.path.join(output_folder, "statistical_analysis.txt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# File mappings for custom labels
file_labels = {
    "NNC_test_text.txt": "That as a Conjunction for a Noun",
    "that_adv.txt": "That as an Adverb",
    "that_conjunction.txt": "That as a Conjunction for a Verb",
    "that_determiner.txt": "That as a Singular Determiner",
    "that_pronoun.txt": "That as a Relative Pronoun",
}

# Define a scientific-friendly color palette
histogram_color = "#4C72B0"  # Dark blue
bar_color = "#DD8452"        # Muted orange
pie_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#937860"]  # Soft, contrasting colors

# Function to analyze a file
def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
    sentence_lengths = [len(s.split()) for s in sentences]  # Count words per sentence

    return len(sentences), sentence_lengths

# Store results
sentence_counts = {}
sentence_lengths = {}

for file, label in file_labels.items():
    filepath = os.path.join(data_folder, file)
    
    if os.path.exists(filepath):  # Check if file exists before processing
        num_sentences, lengths = analyze_file(filepath)
        sentence_counts[label] = num_sentences
        sentence_lengths[label] = lengths

        # Remove "That as a " or "That as an " from label for display purposes
        display_label = label.replace("That as a ", "").replace("That as an ", "")
        
        # Plot histogram for sentence lengths
        plt.figure(figsize=(8, 5))
        plt.hist(lengths, bins=10, color=histogram_color, edgecolor="black", alpha=0.8)
        plt.xlabel("Sentence Length (words)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.title(f"Sentence Length Distribution\n{display_label}", fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{display_label.replace(' ', '_')}_histogram.png"), dpi=300)
        plt.close()

# Save statistical analysis
with open(output_file, "w") as f:
    for label, count in sentence_counts.items():
        avg_length = np.mean(sentence_lengths[label])
        f.write(f"Category: {label}\n")
        f.write(f"Number of sentences: {count}\n")
        f.write(f"Average sentence length: {avg_length:.2f} words\n")
        f.write("-" * 40 + "\n")

# Compute average sentence lengths
avg_lengths = {label: np.mean(lengths) for label, lengths in sentence_lengths.items()}
# Create a dictionary with display labels by removing the prefix
display_avg_lengths = {label.replace("That as a ", "").replace("That as an ", ""): avg for label, avg in avg_lengths.items()}

# Plot comparison of average sentence lengths
plt.figure(figsize=(8, 5))
plt.bar(display_avg_lengths.keys(), display_avg_lengths.values(), color=bar_color, edgecolor="black", alpha=0.8)
plt.xlabel("Categories", fontsize=10)
plt.ylabel("Average Sentence Length (words)", fontsize=10)
plt.title('Comparison of Average Sentence Lengths of "That" as a ... ', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='y', linestyle="--", linewidth=0.5)
plt.tight_layout()  # Ensure labels are not cut off
plt.savefig(os.path.join(output_folder, "average_sentence_length_comparison.png"), dpi=300)
plt.close()

# Plot pie chart for the proportion of average sentence lengths
plt.figure(figsize=(8, 5))
plt.pie(display_avg_lengths.values(), labels=display_avg_lengths.keys(), autopct="%1.1f%%", startangle=140, colors=pie_colors)
plt.title(f'Proportion of Average Sentence Lengths of "That" as a ... ', fontsize=12)
plt.tight_layout()  # Ensure the plot fits well within the figure area
plt.savefig(os.path.join(output_folder, "average_sentence_length_pie_chart.png"), dpi=300)
plt.close()
