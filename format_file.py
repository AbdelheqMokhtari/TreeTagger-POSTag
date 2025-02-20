import os

def transform_file(input_file, output_dir):
    # Extract the base file name (without path and extension) to create a formatted file name
    base_name = os.path.basename(input_file).replace('.txt', '_formatted.txt')
    output_file = os.path.join(output_dir, base_name)  # Full path for the output file

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into words and tags
            tokens_tags = line.strip().split()
            if not tokens_tags:
                continue

            # Check if we have an even number of tokens and tags
            if len(tokens_tags) % 2 != 0:
                print(f"Skipping malformed line (odd number of elements): {line.strip()}")
                continue

            # For each word and tag, write to the output file with a tab
            for i in range(0, len(tokens_tags), 2):  # Increment by 2 to access token/tag pairs
                token = tokens_tags[i]
                tag = tokens_tags[i + 1]
                outfile.write(f"{token}\t{tag}\n")
            # Add a blank line between sentences
            outfile.write("\n")

    print(f"Output written to: {output_file}")


def process_all_files(training_files, output_dir):
    # Ensure the output directory exists, create if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    for input_file in training_files:
        transform_file(input_file, output_dir)


# List of input training files
training_files = [
    'dataset/train_files_claws8/adverb.txt',
    'dataset/train_files_claws8/verb_conjunction.txt',
    'dataset/train_files_claws8/noun_conjunction.txt',
    'dataset/train_files_claws8/determiner.txt',
    'dataset/train_files_claws8/pronoun.txt'
]

# Path for the output formatted files
output_dir = 'dataset/formatted_train_files_claws8/'

# Process all files
process_all_files(training_files, output_dir)
