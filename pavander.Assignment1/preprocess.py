# Paul Vander Woude (pavander) EECS 486 HW1 preprocess.py
import re

def removeSGML(text):
    return re.sub(r"<[^>]+>", "", text)

def tokenizeText(input):
    pass

def BPE(tokens, vocabSize):
    pass

def main():
    pass

if __name__ == "__main__":
    main()