import treetaggerwrapper


# Initialize the TreeTagger instance
tagger = treetaggerwrapper.TreeTagger(TAGPARFILE="/home/abdelhaq/treetagger/lib/english-bnc.par")


# Sample text to tag
text = "I can't believe that happened"

# Get the tagged output
tags = tagger.tag_text(text)

print("\n")
# Print the tagged words, POS, and lemma
for tag in tags:
    word, pos, lemma = tag.split('\t')
    print(f"{word}\t{pos}\t{lemma}")
