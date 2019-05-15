import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer 
import re
import numpy as np

def tokenize(text):
    """Tokenizes and lemmatizes text.

    Args:
        text: Text to be processed.

    Returns:
        The processed text.

    """
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token = re.sub(r'\[[^.,;:]]*\]', '', token)
        
        # add token to compiled list if not empty
        if token != '':
            processed_tokens.append(token)
        
    return processed_tokens

def compute_text_length(data):
    """Calculates the length for each string in a list of string.

    Args:
        data: Data to be analyzed.

    Returns:
        List of lengths corresponding to data.

    """
    return np.array([len(text) for text in data]).reshape(-1, 1)
