import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def init_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def filter_products(df, required_products):
    return df[df['Product'].isin(required_products.values())].copy()

def drop_missing_narratives(df):
    return df[df['Consumer complaint narrative'].notna()].copy()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    init_nltk()
    text = text.lower()
    text = re.sub(r'i am writing to (file|submit|make)?\s*(a)?\s*complaint( about)?', '', text)
    text = re.sub(r'this complaint is regarding', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)