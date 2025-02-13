import treetaggerwrapper


# Initialize the TreeTagger instance
tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')

# Sample text to tag
text = "Hello world! This is a test sentence."

# Get the tagged output
tags = tagger.tag_text(text)

print("\n")
# Print the tagged words, POS, and lemma
for tag in tags:
    word, pos, lemma = tag.split('\t')
    print(f"{word}\t{pos}\t{lemma}")
