import os
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define folder paths
data_folder = "Data/Test"
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

# Function to analyze a file at the sentence level
def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
    sentence_lengths = [len(s.split()) for s in sentences]  # Count words per sentence

    return len(sentences), sentence_lengths

# Containers for sentence-level results
sentence_counts = {}
sentence_lengths = {}

# -------------------------------
# Sentence-level Analysis
# -------------------------------
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

# Save sentence-level statistical analysis results
with open(output_file, "w") as f:
    for label, count in sentence_counts.items():
        avg_length = np.mean(sentence_lengths[label])
        f.write(f"Category: {label}\n")
        f.write(f"Number of sentences: {count}\n")
        f.write(f"Average sentence length: {avg_length:.2f} words\n")
        f.write("-" * 40 + "\n")

# Compute average sentence lengths for each category
avg_lengths = {label: np.mean(lengths) for label, lengths in sentence_lengths.items()}
display_avg_lengths = {label.replace("That as a ", "").replace("That as an ", ""): avg for label, avg in avg_lengths.items()}

# Plot comparison of average sentence lengths
plt.figure(figsize=(8, 5))
plt.bar(display_avg_lengths.keys(), display_avg_lengths.values(), color=bar_color, edgecolor="black", alpha=0.8)
plt.xlabel("Categories", fontsize=10)
plt.ylabel("Average Sentence Length (words)", fontsize=10)
plt.title("Comparison of Average Sentence Lengths", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.grid(axis='y', linestyle="--", linewidth=0.5)
plt.tight_layout()  # Ensure labels are not cut off
plt.savefig(os.path.join(output_folder, "average_sentence_length_comparison.png"), dpi=300)
plt.close()

# Plot pie chart for the proportion of average sentence lengths
plt.figure(figsize=(8, 5))
plt.pie(display_avg_lengths.values(), labels=display_avg_lengths.keys(), autopct="%1.1f%%", startangle=140, colors=pie_colors)
plt.title("Proportion of Average Sentence Lengths", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "average_sentence_length_pie_chart.png"), dpi=300)
plt.close()

# -------------------------------
# Further Analysis: Word Frequency & Lexical Diversity
# -------------------------------

# Containers for further analysis results
word_counts = {}
lexical_diversity = {}
top_words = {}

# Loop through each file to analyze word-level metrics
for file, label in file_labels.items():
    filepath = os.path.join(data_folder, file)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Remove punctuation and convert to lowercase
        translator = str.maketrans("", "", string.punctuation)
        text_clean = text.translate(translator).lower()
        words = text_clean.split()
        # Remove stopwords before further analysis
        words = [w for w in words if w not in stop_words]
        
        total_words = len(words)
        unique_words = len(set(words))
        lexical_diversity[label] = unique_words / total_words if total_words > 0 else 0
        
        # Count word frequencies and get top 10 words
        counts = Counter(words)
        top_words[label] = counts.most_common(10)
        word_counts[label] = counts

        # Plot top 10 words for this file using the same histogram color
        display_label = label.replace("That as a ", "").replace("That as an ", "")
        if top_words[label]:
            words_top, freqs_top = zip(*top_words[label])
            plt.figure(figsize=(8, 5))
            plt.bar(words_top, freqs_top, color=bar_color, edgecolor="black", alpha=0.8)
            plt.xlabel("Words", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.title(f"Top 10 Co-occurring Words with 'That' as {display_label}", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"top_10_words_{display_label.replace(' ', '_')}.png"), dpi=300)
            plt.close()

# Plot comparison of lexical diversity across categories
display_lex_diversity = {label.replace("That as a ", "").replace("That as an ", ""): diversity 
                         for label, diversity in lexical_diversity.items()}

plt.figure(figsize=(8, 5))
plt.bar(display_lex_diversity.keys(), display_lex_diversity.values(), color=bar_color, edgecolor="black", alpha=0.8)
plt.xlabel("Categories", fontsize=10)
plt.ylabel("Lexical Diversity (Unique/Total words)", fontsize=10)
plt.title('Lexical Diversity Comparison in sentences where "That" as a ... ', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "lexical_diversity_comparison.png"), dpi=300)
plt.close()

# Append further analysis results to the text report
with open(output_file, "a") as f:
    f.write("\nFurther Analysis:\n")
    f.write("=" * 40 + "\n")
    for label in file_labels.values():
        f.write(f"{label}:\n")
        f.write(f"Lexical Diversity (Unique/Total words): {lexical_diversity[label]:.2f}\n")
        f.write("Top 10 words:\n")
        for word, freq in top_words[label]:
            f.write(f"  {word}: {freq}\n")
        f.write("-" * 40 + "\n")

print("Analysis completed! All results have been saved in the 'statistical_results' folder.")
