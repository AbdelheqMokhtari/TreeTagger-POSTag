import os
import json
import glob
import treetaggerwrapper
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define folder paths
data_folder = "Data/Test"
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Define file configurations for the test files.
file_configs = [
    {
        "id": "NNC_test_text", 
        "expected_label": "CST", 
        "filename": "NNC_test_text.txt", 
        "filepath": os.path.join(data_folder, "NNC_test_text.txt")
    },
    {
        "id": "that_adv", 
        "expected_label": "RA", 
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
        "expected_label": "DD1", 
        "filename": "that_determiner.txt", 
        "filepath": os.path.join(data_folder, "that_determiner.txt")
    },
    {
        "id": "that_pronoun", 
        "expected_label": "WPR", 
        "filename": "that_pronoun.txt", 
        "filepath": os.path.join(data_folder, "that_pronoun.txt")
    }
]

# Find all .par model files in the Training folder
model_files = glob.glob("Training/*.par")

# Loop over each model file found
for model_path in model_files:
    # Extract model name without extension
    model_name = os.path.basename(model_path).replace(".par", "")
    print(f"Evaluating model: {model_name}")

    # Initialize TreeTagger with the current model
    tagger = treetaggerwrapper.TreeTagger(TAGPARFILE=model_path)

    # Define a helper function that uses the current tagger to get the tag for "that"
    def get_that_tag(sentence):
        tags = tagger.tag_text(sentence)
        for tag in tags:
            parts = tag.split("\t")
            if len(parts) >= 2 and parts[0].lower() == "that":
                return parts[1]  # Return the POS tag
        return None  # "that" not found

    # Initialize containers for overall metrics for the current model
    overall_true = []
    overall_pred = []
    accuracies = {}       
    file_results_dict = {}  
    conf_matrix = {}  # for per-file confusion counts

    # Process each test file defined in file_configs
    for config in file_configs:
        file_id = config["id"]
        expected_label = config["expected_label"]
        file_path = config["filepath"]

        file_results = []  # list to store (sentence, expected, predicted)
        correct_predictions = 0
        total_sentences = 0
        conf_matrix[file_id] = {}  # initialize confusion counts for this file

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    sentence = line.strip()
                    if not sentence:
                        continue  # skip blank lines
                    total_sentences += 1
                    predicted_tag = get_that_tag(sentence)

                    # Collect overall true and predicted labels
                    overall_true.append(expected_label)
                    overall_pred.append(predicted_tag)

                    # Save per-sentence result
                    file_results.append((sentence, expected_label, predicted_tag))

                    # Count correct predictions
                    if predicted_tag == expected_label:
                        correct_predictions += 1

                    # Update per-file confusion matrix
                    conf_matrix[file_id][predicted_tag] = conf_matrix[file_id].get(predicted_tag, 0) + 1

            # Calculate and store accuracy for the file
            accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
            accuracies[file_id] = {"accuracy": accuracy, "num_sentences": total_sentences}
            file_results_dict[file_id] = file_results

            # Save per-file results (with model name prefix)
            output_file_path = os.path.join(output_folder, f"{model_name}_results_{file_id}.txt")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write("Sentence | True Label | Predicted Tag\n")
                output_file.write("-" * 60 + "\n")
                for sentence, true_lab, pred in file_results:
                    output_file.write(f"{sentence} | {true_lab} | {pred}\n")
            print(f"Results for {file_id} saved in {output_file_path}")
        else:
            print(f"File {file_path} not found.")

    # Create a complete confusion matrix DataFrame using file IDs as rows and predicted tags as columns
    all_pred_tags = set()
    for row in conf_matrix.values():
        all_pred_tags.update(row.keys())
    all_pred_tags = sorted(list(all_pred_tags))

    conf_matrix_complete = {}
    for file_id in conf_matrix:
        conf_matrix_complete[file_id] = {tag: conf_matrix[file_id].get(tag, 0) for tag in all_pred_tags}

    conf_matrix_df = pd.DataFrame(conf_matrix_complete).T
    conf_matrix_path = os.path.join(output_folder, f"{model_name}_confusion_matrix.csv")
    conf_matrix_df.to_csv(conf_matrix_path)
    print(f"Confusion Matrix saved in {conf_matrix_path}")

    # --- NEW: Generate a second confusion matrix based on overall true vs. predicted labels ---
    all_possible_tags = ["CST", "RA", "CJT", "DD1", "WPR"]
    cm2 = pd.crosstab(
        pd.Series(overall_true, name='True'),
        pd.Series(overall_pred, name='Predicted'),
        dropna=False
    )
    # Ensure all tags appear as rows and columns even if missing in the data
    for tag in all_possible_tags:
        if tag not in cm2.index:
            cm2.loc[tag] = 0
        if tag not in cm2.columns:
            cm2[tag] = 0

    cm2 = cm2.reindex(index=all_possible_tags, columns=all_possible_tags, fill_value=0)
    cm2_path = os.path.join(output_folder, f"{model_name}_confusion_matrix_2.csv")
    cm2.to_csv(cm2_path)
    print(f"Confusion Matrix 2 saved in {cm2_path}")

    # Generate and save the classification report (precision, recall, f1-score, support)
    class_report = classification_report(overall_true, overall_pred, output_dict=True)
    classification_report_path = os.path.join(output_folder, f"{model_name}_classification_report.json")
    with open(classification_report_path, "w", encoding="utf-8") as json_file:
        json.dump(class_report, json_file, indent=4)
    print(f"Classification Report saved in {classification_report_path}")

    # Save overall accuracy report to JSON
    accuracy_report_path = os.path.join(output_folder, f"{model_name}_accuracy_report.json")
    with open(accuracy_report_path, "w", encoding="utf-8") as json_file:
        json.dump(accuracies, json_file, indent=4)
    print(f"Accuracy Report saved in {accuracy_report_path}")
