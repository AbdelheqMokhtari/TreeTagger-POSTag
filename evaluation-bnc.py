import os
import treetaggerwrapper
import pandas as pd

# Define folder paths
data_folder = "Data"
output_folder = "Results"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define file paths
file_paths = {
    "CJT" : os.path.join(data_folder, "NNC_test_text.txt"), # as a Conjunction for a Noun (IN in penn)
    "AV0": os.path.join(data_folder, "that_adv.txt"),  # Adverb (RB in Penn)
    "CJT": os.path.join(data_folder, "that_conjunction.txt"),  # as a Conjunction for a verb (IN in Penn)
    "DT0": os.path.join(data_folder, "that_determiner.txt"),  # Determiner (DT in Penn)
    "CJT": os.path.join(data_folder, "that_pronoun.txt")  # Pronoun (WDT in Penn)
}

# Initialize TreeTagger with BNC tagset
tagger = treetaggerwrapper.TreeTagger(TAGPARFILE="/home/abdelhaq/treetagger/lib/english-bnc.par")

# Function to extract the POS tag for "that"
def get_that_tag(sentence):
    tags = tagger.tag_text(sentence)
    for tag in tags:
        parts = tag.split("\t")
        if len(parts) >= 2 and parts[0].lower() == "that":
            return parts[1]  # Extract BNC POS tag
    return None  # "that" not found

# List of possible labels
true_labels = list(file_paths.keys())  
predicted_labels = []

# Dictionary to store accuracy per file
accuracies = {}

# Confusion matrix initialization
conf_matrix = {label: {l: 0 for l in true_labels} for label in true_labels}

# Process each file separately
for true_label, file_path in file_paths.items():
    if os.path.exists(file_path):
        correct_predictions = 0
        total_sentences = 0
        file_results = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                sentence = line.strip()
                predicted_tag = get_that_tag(sentence)

                if predicted_tag:
                    predicted_labels.append(predicted_tag)
                    total_sentences += 1

                    # Count correct predictions
                    if predicted_tag == true_label:
                        correct_predictions += 1

                    # Update confusion matrix if predicted tag exists
                    if predicted_tag in conf_matrix:
                        conf_matrix[true_label][predicted_tag] += 1
                    else:
                        print(f"Warning: Unexpected tag '{predicted_tag}' found in sentence: {sentence}")

                    # Store results
                    file_results.append((sentence, true_label, predicted_tag))

        # Compute accuracy for this category
        accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
        accuracies[true_label] = accuracy

        # Save results per category
        output_file_path = os.path.join(output_folder, f"results_{true_label}_bnc.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("Sentence | True Label | Predicted Tag\n")
            output_file.write("-" * 60 + "\n")
            for sentence, true_label, predicted_tag in file_results:
                output_file.write(f"{sentence} | {true_label} | {predicted_tag}\n")

        print(f"Results for {true_label} saved in {output_file_path}")

# Convert confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix).fillna(0)

# Save confusion matrix
conf_matrix_path = os.path.join(output_folder, "confusion_matrix_bnc.csv")
conf_matrix_df.to_csv(conf_matrix_path)

# Save accuracy report
accuracy_report_path = os.path.join(output_folder, "accuracy_report_bnc.txt")
with open(accuracy_report_path, "w", encoding="utf-8") as acc_file:
    acc_file.write("Category-wise Accuracy:\n")
    acc_file.write("-" * 30 + "\n")
    for label, acc in accuracies.items():
        acc_file.write(f"{label}: {acc:.2f}%\n")

print("\nConfusion Matrix and Accuracy Report saved successfully!")