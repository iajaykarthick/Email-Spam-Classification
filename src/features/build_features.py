import re
import pandas as pd
from collections import Counter

import nltk
nltk.download('punkt')


def extract_features(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Create a frequency distribution of the words
    word_freq = nltk.FreqDist(words)

    # Define the words and characters we're interested in
    words_of_interest = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference']
    chars_of_interest = [';', '(', '[', '!', '$', '#']

    # Extract the word and character frequencies
    word_freqs = {f'word_freq_{word}': word_freq[word] for word in words_of_interest}
    char_freqs = {f'char_freq_{char}': text.count(char) for char in chars_of_interest}

    # Extract the capital run length features
    capital_runs = re.findall(r'[A-Z]+', text)
    capital_run_lengths = [len(run) for run in capital_runs]
    capital_run_length_average = sum(capital_run_lengths) / len(capital_run_lengths) if capital_run_lengths else 0
    capital_run_length_longest = max(capital_run_lengths) if capital_run_lengths else 0
    capital_run_length_total = sum(capital_run_lengths)

    # Combine all the features into one dictionary
    features = {**word_freqs, **char_freqs, 'capital_run_length_average': capital_run_length_average, 'capital_run_length_longest': capital_run_length_longest, 'capital_run_length_total': capital_run_length_total}

    return features