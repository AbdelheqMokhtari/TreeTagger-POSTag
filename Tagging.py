import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK data packages
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")  # use "averaged_perceptron_tagger_eng" if needed
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map Treebank tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def map_tag(nltk_tag):
    """Map NLTK tags to your custom tags."""
    mapping = {
        "CC": "CC",      
        "CD": "CD",      
        "DT": "AT",      
        "EX": "EX",      
        "FW": "FW",      
        "IN": "IN",      
        "JJ": "JJ",      
        "JJR": "JJ",     
        "JJS": "JJ",     
        "LS": "LS",      
        "MD": "MD",      
        "NN": "NN",      
        "NNS": "NNS",    
        "NNP": "NN",     
        "NNPS": "NNS",   
        "PDT": "PDT",    
        "POS": "POS",    
        "PRP": "PRP",    
        "PRP$": "PRP",   
        "RB": "RB",      
        "RBR": "RB",     
        "RBS": "RB",     
        "RP": "RP",      
        "SYM": "SYM",    
        "TO": "TO",      
        "UH": "UH",      
        "VB": "VV0",     
        "VBD": "VVD",    
        "VBG": "VV0",    
        "VBN": "VV0",    
        "VBP": "VV0",    
        "VBZ": "VV0",    
        "WDT": "WDT",    
        "WP": "PRP",     
        "WP$": "PRP",    
        "WRB": "RB"
    }
    return mapping.get(nltk_tag, nltk_tag)

def process_file(input_filename, output_filename, that_override_tag, lexicon):
    """
    Process a file: read it line by line (each line as a sentence), tokenize and tag the text,
    override "that" tags as needed, compute the lemma for each word, and update the lexicon dictionary.
    Returns the set of custom tags used in this file.
    """
    with open(input_filename, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # Treat each non-empty line as a sentence.
    sentences = [line.strip() for line in lines if line.strip()]
    processed_lines = []
    file_tags = set()
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for word, tag in tagged:
            # Override the tag for "that" (case insensitive)
            if word.lower() == "that":
                custom_tag = that_override_tag
            else:
                custom_tag = map_tag(tag)
            # Compute lemma using WordNet lemmatizer
            lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
            processed_lines.append(f"{word}\t{custom_tag}")
            file_tags.add(custom_tag)
            # Update lexicon: add (custom_tag, lemma) for the word.
            if word in lexicon:
                lexicon[word].add((custom_tag, lemma))
            else:
                lexicon[word] = {(custom_tag, lemma)}
        # Append one empty string for a blank line between sentences.
        processed_lines.append("")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(processed_lines))
    return file_tags

# Define input and output directories
input_dir = "Data/Train/"
output_dir = "Training/"
os.makedirs(output_dir, exist_ok=True)

all_tags = set()
global_lexicon = {}

# List of files to process: (input_filename, output_filename, override tag)
files_to_process = [
    ("that_as_adverb.txt", "adverb_formatted.txt", "RA"),
    ("that_conjunction_noun.txt", "conjunction_noun_formatted.txt", "CST"),
    ("that_conjunction_verb.txt", "conjunction_verb_formatted.txt", "CJT"),
    ("that_pronoun.txt", "pronoun_formatted.txt", "WPR"),
    ("that_singular_determiner.txt", "determiner_formatted.txt", "DD1")
]

# Process each file, update all_tags and the global lexicon
for infile_name, outfile_name, override in files_to_process:
    input_path = os.path.join(input_dir, infile_name)
    output_path = os.path.join(output_dir, outfile_name)
    tags = process_file(input_path, output_path, override, global_lexicon)
    all_tags.update(tags)

# Write openCLs.txt with all unique tags (sorted)
opencls_path = os.path.join(output_dir, "openCLs.txt")
with open(opencls_path, "w", encoding="utf-8") as tag_file:
    tag_file.write(" ".join(sorted(all_tags)))

# Write lexicon.txt:
# For each word, list the word followed by its tag–lemma pairs (tab separated).
# If a word has multiple pairs, list them all on one line.
# Finally, append a punctuation line: ".	SENT	."
lexicon_path = os.path.join(output_dir, "lexicon.txt")
with open(lexicon_path, "w", encoding="utf-8") as lex_file:
    for word in sorted(global_lexicon.keys(), key=lambda x: x.lower()):
        pairs = sorted(global_lexicon[word])
        pair_strs = []
        for tag, lemma in pairs:
            pair_strs.append(f"{tag}\t{lemma}")
        lex_file.write(f"{word}\t" + "\t".join(pair_strs) + "\n")
    # Append the punctuation entry
    lex_file.write(".\tSENT\t.\n")

# Concatenate the contents of all processed files into train.txt
train_file_path = os.path.join(output_dir, "train.txt")
with open(train_file_path, "w", encoding="utf-8") as train_file:
    for _, outfile_name, _ in files_to_process:
        file_path = os.path.join(output_dir, outfile_name)
        with open(file_path, "r", encoding="utf-8") as infile:
            content = infile.read()
            train_file.write(content)
            train_file.write("\n")  # newline between files

print("Processing complete. Files have been saved in the 'Training/' directory.")
