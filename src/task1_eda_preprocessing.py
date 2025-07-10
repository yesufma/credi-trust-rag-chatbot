import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK
nltk.download('stopwords')
nltk.download('wordnet')

def load_complaints(file_path, chunksize=100000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'i am writing to (file|submit|make)?\s*(a)?\s*complaint( about)?', '', text)
    text = re.sub(r'this complaint is regarding', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def main():
    df = load_complaints('data/raw/complaints.csv')
    required_products = {
        'Credit card': 'Credit card',
        'Personal loan': 'Consumer Loan',
        'Buy Now, Pay Later (BNPL)': 'Payday loan',
        'Savings account': 'Checking or savings account',
        'Money transfers': 'Money transfer, virtual currency, or remittance'
    }
    df_filtered = df[df['Product'].isin(required_products.values())]
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notna()]
    df_filtered['clean_narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    os.makedirs('data/processed', exist_ok=True)
    df_filtered.to_csv('data/processed/filtered_complaints.csv', index=False)
    print("✅ Task 1 completed: Data processed and saved")

if __name__ == "__main__":
    main()
