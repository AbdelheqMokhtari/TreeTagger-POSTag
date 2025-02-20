def extract_unique_tags(lexicon_file, output_file):
    unique_tags = set()  # Using a set to ensure tags are unique

    # Open the lexicon file and read it
    with open(lexicon_file, 'r') as infile:
        for line in infile:
            parts = line.strip().split()  # Split the line into parts
            if len(parts) == 3:  # Ensure the line has exactly three parts: word, tag, and word again
                tag = parts[1]  # The second part is the tag
                unique_tags.add(tag)  # Add the tag to the set

    # Write the unique tags to the output file
    with open(output_file, 'w') as outfile:
        # Sort the tags alphabetically and write them to the output file
        for tag in sorted(unique_tags):
            outfile.write(tag + '\n')

    print(f"Unique tags have been written to: {output_file}")

# Example usage
lexicon_file = 'dataset/lexicon.txt'  # Path to your lexicon file
output_file = 'dataset/OpenCLs.txt'   # Path to the output OpenCLs file

extract_unique_tags(lexicon_file, output_file)
