# Paul Vander Woude (pavander) EECS 486 HW1 languageIdentification.py

import math
import sys
import os
from collections import Counter

def trainBigramLanguageModel(text):
    """Trains a bigram language model by calculating unigram and bigram frequencies."""
    
    # Remove unwanted characters and ensure text is lowercase for consistency
    text = text.lower()
    
    # Tokenize the text into characters
    tokens = list(text)
    
    # Calculate unigram frequencies
    unigram_freq = Counter(tokens)
    
    # Calculate bigram frequencies
    bigram_freq = Counter(zip(tokens[:-1], tokens[1:]))  # Pair consecutive tokens
    
    return dict(unigram_freq), dict(bigram_freq)

def calculate_bigram_probability(text, unigram_freq, bigram_freq):
    """Calculates the bigram probability of a given text using unigram and bigram frequencies."""
    
    # Tokenize the text into characters
    tokens = list(text.lower())
    
    # Initialize probability
    prob = 0.0
    
    # Loop through the bigrams in the text
    for i in range(1, len(tokens)):
        prev_char = tokens[i - 1]
        curr_char = tokens[i]
        
        # If the bigram exists in the bigram dictionary, calculate its probability
        bigram_count = bigram_freq.get((prev_char, curr_char), 0)
        unigram_count = unigram_freq.get(prev_char, 0)
        
        # add laplace smoothing to avoid zero probability
        if unigram_count > 0:
            prob += math.log((bigram_count + 1) / (unigram_count + len(unigram_freq)))
        else:
            prob += math.log(1 / (len(unigram_freq) + len(unigram_freq)))  # Add small value to avoid zero
    
    return prob

def identifyLanguage(text, language_names, unigram_char_freq_dicts, bigram_char_freq_dicts):
    """Identifies the language of the input text based on bigram probabilities."""
    
    # Initialize variables to track the best language and highest probability
    best_language = None
    highest_prob = float('-inf')
    
    # Loop through the language names to calculate bigram probabilities
    for language in language_names:
        unigram_freq = unigram_char_freq_dicts[language]
        bigram_freq = bigram_char_freq_dicts[language]
        
        # Calculate the bigram probability of the input text for this language
        prob = calculate_bigram_probability(text, unigram_freq, bigram_freq)
        
        # Update the best language if the current language has a higher probability
        if prob > highest_prob:
            highest_prob = prob
            best_language = language
    
    return best_language


def main():
    training_dir = sys.argv[1]
    test_file_path = sys.argv[2]

    language_names = []
    uni_freq_by_lang, bi_freq_by_lang = {}

    # Step 1: Training
    if not os.path.isdir(training_dir):
        raise TypeError(f'USAGE: python3 languageIdentification.py TRAINING_DIR TEST_FILE_PATH\n{training_dir} is not a directory')
    
    if not os.path.isfile(test_file_path):
        raise TypeError(f'USAGE: python3 languageIdentification.py TRAINING_DIR TEST_FILE_PATH\n{test_file_path} is not a file')
    
    for filename in os.listdir(training_dir):
        language_names.append(filename)

        file_path = '/'.join(training_dir, filename)

        with open(file_path, "r", encoding="ISO-8859-1") as file:
            text = file.read()

            unigram, bigram = trainBigramLanguageModel(text)

            uni_freq_by_lang[filename] = unigram
            bi_freq_by_lang[filename] = bigram
    
    with open(test_file_path, "r", encoding="ISO-8859-1") as in_file:
        with open("languageIdentification.output", "w", encoding="ISO-8859-1") as out_file:
            lineCounter = 0
            for in_line in in_file:
                lineCounter += 1
                lang = identifyLanguage(in_line, language_names, uni_freq_by_lang, bi_freq_by_lang)
                out_line = f"{lineCounter} {lang}\n"
                out_file.write(out_line)


if __name__ == "__main__":
    main()