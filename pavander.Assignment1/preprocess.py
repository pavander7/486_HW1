# Paul Vander Woude (pavander) EECS 486 HW1 preprocess.py
import os
import sys
import re
from collections import Counter

# Dictionary for expanding contractions
CONTRACTIONS = {
    "can't": ["can", "not"], 
    "won't": ["will", "not"], 
    "n't": ["not"], 
    "'re": ["are"], 
    "'s": ["'s"], 
    "'ll": ["will"], 
    "'d": ["would"], 
    "'ve": ["have"], 
    "'m": ["am"]
}

# PART 1: removeSGML

def removeSGML(text):
    """Cleans raw text by removing SGML tags."""
    return re.sub(r"<[^>]+>", "", text)


# PART 2: tokenizeText

def expand_contractions(token):
    """Expands contractions based on the CONTRACTIONS dictionary."""
    for contraction, expansion in CONTRACTIONS.items():
        if token.lower().endswith(contraction):
            base = token[: -len(contraction)]
            return ([base] if base else []) + expansion
    return [token]

def tokenizeText(text):
    """Tokenizes text that has been stripped of SGML tags."""
    # Regular expression for tokenizing
    pattern = r"""
        \b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b       # Dates (e.g., 01/31/2024, 2024-01-31)
        |\b(?:[A-Za-z]+\.){2,}                    # Acronyms (e.g., U.S.A., E.U.)
        |\b\w+(?:-\w+)+\b                          # Hyphenated words (e.g., mother-in-law)
        |\b\w+\b                                   # Words
        |\d+(?:,\d{3})*(?:\.\d+)?\b                # Numbers with commas/decimals (e.g., 1,000.50)
    """

    # Split the text into lines
    lines = text.splitlines()
    
    all_tokens = []
    
    for line in lines:
        # Tokenize each line, prepending <start> token
        tokens = re.findall(pattern, line, re.VERBOSE)
        
        # Add <start> token at the beginning of each line
        line_tokens = ["<start>"] + tokens
        
        # Process contractions and possessives
        final_tokens = []
        for token in line_tokens:
            if token.endswith("'s"):
                final_tokens.append(token[:-2])  # Remove 's'
                final_tokens.append("'s")        # Keep possessive separately
            else:
                final_tokens.extend(expand_contractions(token))  # Expand contractions

        # Add processed tokens to the final list
        all_tokens.extend(final_tokens)

    return all_tokens


# PART 3: BPE

def calc_char_freqs(tokens):
    """Calculates character frequencies from a list of tokens."""
    # Step 1: Concatenate all tokens into a single string
    dirty_text = ''.join(tokens)

    # Use removeSGML to remove <start> tags
    all_text = removeSGML(dirty_text)
    
    # Step 2: Calculate character frequencies
    char_freq = Counter(all_text)
    
    # Step 3: Create the vocabulary with all characters found
    vocab = {char: freq for char, freq in char_freq.items()}

    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    
    return vocab

from collections import Counter

def find_token_pairs(words, vocab):
    """Finds all token pairs in the token list and calculates frequencies without double-counting characters."""
    token_pairs = []
    
    # Track tokens that have already been merged (not just vocab.keys(), but subunits that have been merged)
    merged_tokens = set(vocab.keys())

    for word in words:
        if word != '<start>' and word not in merged_tokens:
            # Tokenize the word into subunits based on vocab
            subunits = []
            i = 0
            while i < len(word):
                # Try to find the longest subunit match in vocab
                curr_subunit = None
                for length in range(1, len(word) - i + 1):
                    subunit = word[i:i + length]
                    if subunit in merged_tokens:
                        curr_subunit = subunit
                if curr_subunit is None:
                    curr_subunit = word[i]
                    vocab[curr_subunit] = 1
                    print(f'created {curr_subunit}')
                    merged_tokens = set(vocab.keys())
                    i += 1
                else:
                    subunits.append(curr_subunit)
                    i += len(curr_subunit)


            # Create adjacent pairs from subunits
            for i in range(len(subunits) - 1):
                pair = (subunits[i], subunits[i + 1])
                token_pairs.append(pair)


    # Count the frequencies of token pairs
    pair_freq = Counter(token_pairs)
    
    return pair_freq, vocab


def merge_token_pairs(vocab, pair_freq):
    """Merges the most common token pair."""
    # Step 1: Find the most common token pair
    most_common_pair, pair_count = pair_freq.most_common(1)[0]

    # Step 2: Create a new merged token
    merged_token = ''.join(most_common_pair)
    
    # Step 3: Update the vocabulary by adding the merged token
    vocab[merged_token] = pair_count
    vocab[most_common_pair[0]] -= pair_count
    vocab[most_common_pair[1]] -= pair_count
    
    # Remove any tokens that have a frequency of zero
    if vocab[most_common_pair[0]] <= 0:
        print(f'deleted {most_common_pair[0]}')
        del vocab[most_common_pair[0]]
    if most_common_pair[0] != most_common_pair[1] and vocab[most_common_pair[1]] <= 0:
        print(f'deleted {most_common_pair[1]}')
        del vocab[most_common_pair[1]]

    # Step 4: Save the merge rule (which pair was merged into which token)
    merge_rule = f"({most_common_pair[0]},{most_common_pair[1]}) -> {merged_token}"
    
    return vocab, merge_rule

def BPE(tokens, vocabSize):
    """Performs BPE on a list of tokens."""
    iter_counter = 0

    # Step 1: Initialize vocabulary and character frequencies
    vocab = calc_char_freqs(tokens)  # Initialize vocab with whole tokens
    
    # Initialize an empty list to hold the merge rules
    merge_rules = []

    # Step 2: Loop until vocabulary size reaches vocabSize
    while len(vocab) < vocabSize:
        # Find token pair frequencies
        pair_freq, vocab = find_token_pairs(tokens, vocab)

        # Merge the most common token pair
        vocab, new_merge_rule = merge_token_pairs(vocab, pair_freq)

        # Append the merge rule to the list
        merge_rules.append(new_merge_rule)

        iter_counter += 1
        print(f'{iter_counter}: vocab size {len(vocab)}/{vocabSize} ({round((len(vocab)/vocabSize*100), 2)}%)')
        print(new_merge_rule)

    return vocab, merge_rules


# PART 4: main

def main():
    input_dir = sys.argv[1]
    vocabSize = int(sys.argv[2])

    tokens = []

    if not os.path.isdir(input_dir):
        raise TypeError(f'USAGE: python3 preprocess.py INPUT_DIR VOCAB_SIZE\n{input_dir} is not a directory')

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if not os.path.isfile(file_path):
            raise IsADirectoryError(f'directory found at {file_path} (input_dir should contain only files)')
        
        with open(file_path, "r", encoding="ISO-8859-1") as file:
            # read in file contents
            raw_text = file.read()

            # clean text
            clean_text = removeSGML(raw_text)

            # tokenize text
            new_tokens = tokenizeText(clean_text)

            # save list of tokens
            tokens.extend(new_tokens)
        
    # \for

    vocab, merge_rules = BPE(tokens, vocabSize)

    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

    with open("preprocess.output", "w", encoding="ISO-8859-1") as file:
        file.write(f"Tokens:\t{len(vocab)}\n")
        file.write(f"Merge rules:\t{len(merge_rules)}\n")
        file.write(f"The first 20 merge rules:\n")
        for i in range(0,min(21, len(merge_rules))): 
            file.write(f"\t{merge_rules[i]}\n")
        file.write(f"Top 50 Tokens:\n")
        for i, (token, freq) in enumerate(list(vocab.items())[:50]):
            file.write(f"\t{token} [{freq}]\n")





if __name__ == "__main__":
    main()