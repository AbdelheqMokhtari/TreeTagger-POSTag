import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK data packages
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng") 
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

def get_lemma(token, tag):
    """
    Get the lemma of a token using the WordNet lemmatizer.
    The function maps custom (CLAWS8-like) tags to the corresponding WordNet POS.
    """
    if tag.startswith('VV'):
        pos = wordnet.VERB
    elif tag.startswith('JJ'):
        pos = wordnet.ADJ
    elif tag.startswith('NN'):
        pos = wordnet.NOUN
    elif tag.startswith('RB'):
        pos = wordnet.ADV
    elif tag.startswith('PRP') or tag.startswith('PRP$'):
        pos = wordnet.NOUN 
    elif tag.startswith('DT') or tag.startswith('WDT'):
        pos = wordnet.NOUN  
    elif tag.startswith('IN'):
        pos = wordnet.ADV 
    else:
        pos = wordnet.NOUN
    return lemmatizer.lemmatize(token.lower(), pos)

def map_tag(nltk_tag):
    """
    Map NLTK (Penn Treebank) tags to custom CLAWS8-like tags.
    """
    mapping = {
        "CC": "CC",        # Coordinating conjunction
        "CD": "CD",        # Cardinal number
        "DT": "AT0",       # Determiner/article (CLAWS8 uses AT0 for articles)
        "EX": "EX",        # Existential there
        "FW": "FW",        # Foreign word
        "IN": "II",        # Preposition or subordinating conjunction 
        "JJ": "AJ0",       # Adjective, positive degree
        "JJR": "AJC",      # Adjective, comparative
        "JJS": "AJS",      # Adjective, superlative
        "LS": "LS",        # List marker
        "MD": "MD",        # Modal
        "NN": "NN1",       # Singular common noun
        "NNS": "NN2",      # Plural common noun
        "NNP": "NP",       # Singular proper noun
        "NNPS": "NPS",     # Plural proper noun
        "PDT": "DD0",      # Pre-determiner
        "POS": "POS",      # Possessive ending
        "PRP": "PP",       # Personal pronoun
        "PRP$": "PP$",     # Possessive pronoun
        "RB": "RG0",       # Adverb, base form
        "RBR": "RGC",      # Adverb, comparative
        "RBS": "RGS",      # Adverb, superlative
        "RP": "RP",        # Particle
        "SYM": "SYM",      # Symbol
        "TO": "TO",        # To
        "UH": "UH",        # Interjection
        "VB": "VVB",       # Verb, base form
        "VBD": "VVD",      # Verb, past tense
        "VBG": "VVG",      # Verb, present participle/gerund
        "VBN": "VVN",      # Verb, past participle
        "VBP": "VVP",      # Verb, non-3rd person singular present
        "VBZ": "VVZ",      # Verb, 3rd person singular present
        "WDT": "WDT",      # Wh-determiner
        "WP": "WP",        # Wh-pronoun
        "WP$": "WPS",      # Possessive wh-pronoun
        "WRB": "WRB"       # Wh-adverb
    }
    return mapping.get(nltk_tag, nltk_tag)

def process_file(input_filename, output_filename, that_override_tag, lexicon):
    """
    Process a file: for each line (sentence), tokenize and POS-tag the text,
    override "that" tags as needed, compute the lemma using get_lemma, and update the lexicon.
    Returns the set of custom tags used in the file.
    """
    with open(input_filename, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

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
            # Compute lemma using the custom get_lemma function
            lemma = get_lemma(word, custom_tag)
            processed_lines.append(f"{word}\t{custom_tag}")
            file_tags.add(custom_tag)
            # Update the lexicon: add a (custom_tag, lemma) pair for the word.
            if word in lexicon:
                lexicon[word].add((custom_tag, lemma))
            else:
                lexicon[word] = {(custom_tag, lemma)}
        
        processed_lines.append("")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(processed_lines))
    return file_tags

# Define input and output directories.
input_dir = "Data/Train/"
output_dir = "Training/"
os.makedirs(output_dir, exist_ok=True)

all_tags = set()
global_lexicon = {}

# List of files to process
files_to_process = [
    ("that_as_adverb.txt", "adverb_formatted.txt", "RA"),
    ("that_conjunction_noun.txt", "conjunction_noun_formatted.txt", "CST"),
    ("that_conjunction_verb.txt", "conjunction_verb_formatted.txt", "CJT"),
    ("that_pronoun.txt", "pronoun_formatted.txt", "WPR"),
    ("that_singular_determiner.txt", "determiner_formatted.txt", "DD1")
]

# Process each file, updating the set of all custom tags and the global lexicon.
for infile_name, outfile_name, override in files_to_process:
    input_path = os.path.join(input_dir, infile_name)
    output_path = os.path.join(output_dir, outfile_name)
    tags = process_file(input_path, output_path, override, global_lexicon)
    all_tags.update(tags)

# Write openCLs.txt containing all unique tags 
opencls_path = os.path.join(output_dir, "openCLs.txt")
with open(opencls_path, "w", encoding="utf-8") as tag_file:
    tag_file.write(" ".join(sorted(all_tags)))

# Write lexicon.txt:
# Each line contains a word followed by its tag–lemma pairs (tab separated).
# Finally, append a punctuation line.
lexicon_path = os.path.join(output_dir, "lexicon.txt")
with open(lexicon_path, "w", encoding="utf-8") as lex_file:
    for word in sorted(global_lexicon.keys(), key=lambda x: x.lower()):
        pairs = sorted(global_lexicon[word])
        pair_strs = [f"{tag}\t{lemma}" for tag, lemma in pairs]
        lex_file.write(f"{word}\t" + "\t".join(pair_strs) + "\n")
    lex_file.write(".\tSENT\t.\n")

# Concatenate the contents of all processed files into train.txt
train_file_path = os.path.join(output_dir, "train.txt")
with open(train_file_path, "w", encoding="utf-8") as train_file:
    for _, outfile_name, _ in files_to_process:
        file_path = os.path.join(output_dir, outfile_name)
        with open(file_path, "r", encoding="utf-8") as infile:
            content = infile.read()
            train_file.write(content)
            train_file.write("\n") 

print("Processing complete. Files have been saved in the 'Training/' directory.")
