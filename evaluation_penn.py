import os
import json
import treetaggerwrapper
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Define folder paths
data_folder = "Data/Test"
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Define file configurations.
# Note: Even if two files share the same expected tag (e.g. "IN"),
# they get a unique identifier (id) so that their accuracy and results are computed separately.
file_configs = [
    {
        "id": "NNC_test_text", 
        "expected_label": "IN", 
        "filename": "NNC_test_text.txt", 
        "filepath": os.path.join(data_folder, "NNC_test_text.txt")
    },
    {
        "id": "that_adv", 
        "expected_label": "RB", 
        "filename": "that_adv.txt", 
        "filepath": os.path.join(data_folder, "that_adv.txt")
    },
    {
        "id": "that_conjunction", 
        "expected_label": "IN", 
        "filename": "that_conjunction.txt", 
        "filepath": os.path.join(data_folder, "that_conjunction.txt")
    },
    {
        "id": "that_determiner", 
        "expected_label": "DT", 
        "filename": "that_determiner.txt", 
        "filepath": os.path.join(data_folder, "that_determiner.txt")
    },
    {
        "id": "that_pronoun", 
        "expected_label": "WDT", 
        "filename": "that_pronoun.txt", 
        "filepath": os.path.join(data_folder, "that_pronoun.txt")
    }
]

# Initialize TreeTagger for English.
tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")

def get_that_tag(sentence):
    """
    Process a sentence with TreeTagger and return the POS tag for the token "that".
    """
    tags = tagger.tag_text(sentence)
    for tag in tags:
        parts = tag.split("\t")
        if len(parts) >= 2 and parts[0].lower() == "that":
            # Extract main POS tag (it might come as e.g. "IN/some_info", so we take the part before the "/")
            return parts[1].split("/")[0]
    return None  # "that" not found

# Lists to collect overall predictions for classification report.
overall_true = []
overall_pred = []

# Dictionaries to store per-file results.
accuracies = {}     # Will hold both accuracy and sentence count per file.
file_results_dict = {}  # Optional: to store detailed per-sentence results if needed.

# We build a confusion matrix using file IDs as rows.
conf_matrix = {}  # {file_id: {predicted_tag: count}}

# Process each file separately
for config in file_configs:
    file_id = config["id"]
    expected_label = config["expected_label"]
    file_path = config["filepath"]
    
    file_results = []  # To store tuples: (sentence, expected_label, predicted_tag)
    correct_predictions = 0
    total_sentences = 0
    conf_matrix[file_id] = {}  # Initialize confusion matrix row for this file id

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                sentence = line.strip()
                if not sentence:
                    continue  # Skip blank lines
                total_sentences += 1
                predicted_tag = get_that_tag(sentence)
                
                # Append overall true/predicted labels
                overall_true.append(expected_label)
                overall_pred.append(predicted_tag)
                
                file_results.append((sentence, expected_label, predicted_tag))
                
                # Count correct predictions
                if predicted_tag == expected_label:
                    correct_predictions += 1
                
                # Update the per-file confusion matrix (for the current file id)
                if predicted_tag not in conf_matrix[file_id]:
                    conf_matrix[file_id][predicted_tag] = 0
                conf_matrix[file_id][predicted_tag] += 1

        # Compute accuracy for this file
        accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
        accuracies[file_id] = {"accuracy": accuracy, "num_sentences": total_sentences}
        file_results_dict[file_id] = file_results

        # Save per-file results to a text file
        output_file_path = os.path.join(output_folder, f"results_{file_id}.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("Sentence | True Label | Predicted Tag\n")
            output_file.write("-" * 60 + "\n")
            for sentence, true_lab, pred in file_results:
                output_file.write(f"{sentence} | {true_lab} | {pred}\n")
        print(f"Results for {file_id} saved in {output_file_path}")
    else:
        print(f"File {file_path} not found.")

# Create a complete confusion matrix DataFrame.
# We want rows corresponding to file IDs and columns for every predicted tag encountered.
all_pred_tags = set()
for row in conf_matrix.values():
    all_pred_tags.update(row.keys())
all_pred_tags = sorted(list(all_pred_tags))

# Build a complete confusion matrix dictionary
conf_matrix_complete = {}
for file_id in conf_matrix:
    conf_matrix_complete[file_id] = {tag: conf_matrix[file_id].get(tag, 0) for tag in all_pred_tags}

conf_matrix_df = pd.DataFrame(conf_matrix_complete).T
conf_matrix_path = os.path.join(output_folder, "confusion_matrix.csv")
conf_matrix_df.to_csv(conf_matrix_path)
print(f"Confusion Matrix saved in {conf_matrix_path}")

# --- NEW: Generate Confusion Matrix 2 based on True Labels vs. Predicted Labels ---
# Here, rows represent true labels and columns represent predicted labels.
cm2 = pd.crosstab(pd.Series(overall_true, name='True'), pd.Series(overall_pred, name='Predicted'))
cm2_path = os.path.join(output_folder, "confusion_matrix_2.csv")
cm2.to_csv(cm2_path)
print(f"Confusion Matrix 2 saved in {cm2_path}")

# Generate classification report (includes recall, precision, f1, and support)
class_report = classification_report(overall_true, overall_pred, output_dict=True)
classification_report_path = os.path.join(output_folder, "classification_report.json")
with open(classification_report_path, "w", encoding="utf-8") as json_file:
    json.dump(class_report, json_file, indent=4)
print(f"Classification Report saved in {classification_report_path}")

# Save overall accuracy report (including number of sentences per file) to JSON
accuracy_report_path = os.path.join(output_folder, "accuracy_report.json")
with open(accuracy_report_path, "w", encoding="utf-8") as json_file:
    json.dump(accuracies, json_file, indent=4)
print(f"Accuracy Report saved in {accuracy_report_path}")
