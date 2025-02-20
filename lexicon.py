import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files if not already present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to get the lemma using NLTK's WordNetLemmatizer
def get_lemma(token, tag):
    # Map CLaws8 tags to WordNet POS tags
    if tag.startswith('VB'):
        pos = wordnet.VERB
    elif tag.startswith('JJ'):
        pos = wordnet.ADJ
    elif tag.startswith('NN'):
        pos = wordnet.NOUN
    elif tag.startswith('RB'):
        pos = wordnet.ADV
    elif tag.startswith('PRP') or tag.startswith('PRP$'):
        pos = wordnet.NOUN  # Treat personal and possessive pronouns as nouns
    elif tag.startswith('DT') or tag.startswith('WDT'):
        pos = wordnet.NOUN  # Treat determiners as nouns
    elif tag.startswith('IN'):
        pos = wordnet.ADV  # Treat prepositions and conjunctions as adverbs
    else:
        pos = wordnet.NOUN  # Default to noun if tag is unknown
    
    # Lemmatize the token based on its POS tag
    return lemmatizer.lemmatize(token.lower(), pos)

# Function to create the lexicon from the training file
def create_lexicon(training_files):
    lexicon = {}

    # Iterate through each training file
    for file_name in training_files:
        with open(file_name, 'r') as file:
            for line in file:
                # Split each line into token and POS tag
                tokens_tags = line.strip().split()
                
                if len(tokens_tags) < 2:
                    continue
                token = tokens_tags[0]
                print(token)
                tag = tokens_tags[1]
                print(tag)
                # Get the lemma of the token using NLTK
                lemma = get_lemma(token, tag)

                # Add to the lexicon dictionary (handling duplicates)
                if token not in lexicon:
                    lexicon[token] = set()  # Use a set to store unique tags for the token
                lexicon[token].add(f"{tag} {lemma}")

    return lexicon

# Function to write the lexicon to a file
def write_lexicon_to_file(lexicon, output_file):
    with open(output_file, 'w') as file:
        for token, tags in lexicon.items():
            for tag_lemma in tags:
                file.write(f"{token} {tag_lemma}\n")

# List of your training files (modify with your actual filenames)
training_files = [
    'dataset/formatted_train_files/adverb_formatted.txt',
    'dataset/formatted_train_files/conjuction_formatted.txt',
    'dataset/formatted_train_files/determiner_formatted.txt',
    'dataset/formatted_train_files/pronoun_formatted.txt'
]

# Output lexicon file name
output_file = 'dataset/lexicon.txt'

# Generate the lexicon
lexicon = create_lexicon(training_files)

# Write the lexicon to a file
write_lexicon_to_file(lexicon, output_file)

print(f"Lexicon file '{output_file}' has been created successfully!")
