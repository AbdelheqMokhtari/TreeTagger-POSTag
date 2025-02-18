import os
import json
import treetaggerwrapper
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Define folder paths
data_folder = "Data"
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Define file configurations for the BNC model.
# Note: The expected labels for conjunction uses remain "CJT".
file_configs = [
    {
        "id": "NNC_test_text", 
        "expected_label": "CJT", 
        "filename": "NNC_test_text.txt", 
        "filepath": os.path.join(data_folder, "NNC_test_text.txt")
    },
    {
        "id": "that_adv", 
        "expected_label": "AV0", 
        "filename": "that_adv.txt", 
        "filepath": os.path.join(data_folder, "that_adv.txt")
    },
    {
        "id": "that_conjunction", 
        "expected_label": "CJT", 
        "filename": "that_conjunction.txt", 
        "filepath": os.path.join(data_folder, "that_conjunction.txt")
    },
    {
        "id": "that_determiner", 
        "expected_label": "DT0", 
        "filename": "that_determiner.txt", 
        "filepath": os.path.join(data_folder, "that_determiner.txt")
    },
    {
        "id": "that_pronoun", 
        "expected_label": "CJT", 
        "filename": "that_pronoun.txt", 
        "filepath": os.path.join(data_folder, "that_pronoun.txt")
    }
]

# Initialize TreeTagger with BNC tagset
tagger = treetaggerwrapper.TreeTagger(TAGPARFILE="/home/abdelhaq/treetagger/lib/english-bnc.par")

def get_that_tag(sentence):
    """
    Process a sentence with TreeTagger (BNC model) and return the POS tag for the token "that".
    """
    tags = tagger.tag_text(sentence)
    for tag in tags:
        parts = tag.split("\t")
        if len(parts) >= 2 and parts[0].lower() == "that":
            return parts[1]  # Return the BNC POS tag
    return None  # "that" not found

# Lists for overall true and predicted labels (for the classification report)
overall_true = []
overall_pred = []

# Dictionaries for per-file results
accuracies = {}       # Will store accuracy and number of sentences per file
file_results_dict = {}  # Detailed per-sentence results (if needed)

# Build a confusion matrix using file IDs as rows (for per-file results)
conf_matrix = {}  # {file_id: {predicted_tag: count}}

# Process each file separately
for config in file_configs:
    file_id = config["id"]
    expected_label = config["expected_label"]
    file_path = config["filepath"]
    
    file_results = []  # To store tuples: (sentence, expected_label, predicted_tag)
    correct_predictions = 0
    total_sentences = 0
    conf_matrix[file_id] = {}  # Initialize row for this file
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                sentence = line.strip()
                if not sentence:
                    continue  # Skip blank lines
                total_sentences += 1
                predicted_tag = get_that_tag(sentence)
                
                # Append overall true and predicted labels
                overall_true.append(expected_label)
                overall_pred.append(predicted_tag)
                
                # Save the per-sentence result
                file_results.append((sentence, expected_label, predicted_tag))
                
                # Count correct predictions
                if predicted_tag == expected_label:
                    correct_predictions += 1
                
                # Update the per-file confusion matrix
                if predicted_tag not in conf_matrix[file_id]:
                    conf_matrix[file_id][predicted_tag] = 0
                conf_matrix[file_id][predicted_tag] += 1

        # Compute accuracy for this file
        accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
        accuracies[file_id] = {"accuracy": accuracy, "num_sentences": total_sentences}
        file_results_dict[file_id] = file_results

        # Save per-file results to a text file with a bnc_ prefix
        output_file_path = os.path.join(output_folder, f"bnc_results_{file_id}.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("Sentence | True Label | Predicted Tag\n")
            output_file.write("-" * 60 + "\n")
            for sentence, true_lab, pred in file_results:
                output_file.write(f"{sentence} | {true_lab} | {pred}\n")

        print(f"Results for {file_id} saved in {output_file_path}")
    else:
        print(f"File {file_path} not found.")

# Create the complete confusion matrix DataFrame (rows: file IDs; columns: predicted tags)
all_pred_tags = set()
for row in conf_matrix.values():
    all_pred_tags.update(row.keys())
all_pred_tags = sorted(list(all_pred_tags))

conf_matrix_complete = {}
for file_id in conf_matrix:
    conf_matrix_complete[file_id] = {tag: conf_matrix[file_id].get(tag, 0) for tag in all_pred_tags}

conf_matrix_df = pd.DataFrame(conf_matrix_complete).T
conf_matrix_path = os.path.join(output_folder, "bnc_confusion_matrix.csv")
conf_matrix_df.to_csv(conf_matrix_path)
print(f"Confusion Matrix saved in {conf_matrix_path}")

# --- NEW: Generate Confusion Matrix 2 based on True vs. Predicted Labels ---
# Ensure rows and columns always include "AV0", "CJT", and "DT0"
all_possible_tags = ["AV0", "CJT", "DT0"]

# Create the confusion matrix using pd.crosstab
cm2 = pd.crosstab(
    pd.Series(overall_true, name='True'),
    pd.Series(overall_pred, name='Predicted'),
    dropna=False
)

# Ensure all tags appear as rows and columns (even if missing in data)
for tag in all_possible_tags:
    if tag not in cm2.index:
        cm2.loc[tag] = 0  # Add missing row
    if tag not in cm2.columns:
        cm2[tag] = 0  # Add missing column

# Reorder rows and columns to follow all_possible_tags order
cm2 = cm2.reindex(index=all_possible_tags, columns=all_possible_tags, fill_value=0)

# Save the modified confusion matrix
cm2_path = os.path.join(output_folder, "bnc_confusion_matrix_2.csv")
cm2.to_csv(cm2_path)
print(f"Confusion Matrix 2 saved in {cm2_path}")


# Generate classification report (includes recall, precision, f1-score, and support)
class_report = classification_report(overall_true, overall_pred, output_dict=True)
classification_report_path = os.path.join(output_folder, "bnc_classification_report.json")
with open(classification_report_path, "w", encoding="utf-8") as json_file:
    json.dump(class_report, json_file, indent=4)
print(f"Classification Report saved in {classification_report_path}")

# Save overall accuracy report (including number of sentences per file) to JSON
accuracy_report_path = os.path.join(output_folder, "bnc_accuracy_report.json")
with open(accuracy_report_path, "w", encoding="utf-8") as json_file:
    json.dump(accuracies, json_file, indent=4)
print(f"Accuracy Report saved in {accuracy_report_path}")
