# 🔍 CrediTrust Financial - RAG-Powered Complaint Analysis Chatbot

<div align="center">
  
*Transforming customer complaints into strategic business intelligence*

</div>

## 📌 Project Overview
This project develops an AI-powered chatbot that analyzes customer complaints to provide actionable insights for CrediTrust Financial. Using Retrieval-Augmented Generation (RAG) technology, it enables product managers to identify complaint trends in seconds rather than days.

**Business Impact**:
- ⏱️ **Reduced trend identification** time from days to minutes
- 💡 **Empowered non-technical teams** with self-service insights
- 🔍 **Shifted from reactive to proactive** issue resolution
- 📈 **Improved customer satisfaction** through faster response times

## 🧠 Business Challenge
CrediTrust Financial receives **thousands of monthly complaints** across five product categories:
- 💳 Credit Cards
- 🏦 Personal Loans
- 🛒 Buy Now, Pay Later (BNPL)
- 💰 Savings Accounts
- 🌍 Money Transfers

Product managers like Asha spend hours manually reading complaints to identify issues. Our solution provides instant, evidence-backed answers to questions like:
> "Why are customers unhappy with BNPL?"
> "What's the top complaint for savings accounts in Kenya?"

## 🛠️ Technical Solution
We built a **RAG-powered chatbot** that:
1. **Retrieves** relevant complaints using semantic search (FAISS/ChromaDB)
2. **Generates** insights using large language models (Mistral/Llama)
3. **Presents** answers with source evidence for verification



## 📂 Project Structure
```bash

credi-trust-rag-chatbot/
├── data/                   # Processed complaint data
│   └── filtered_complaints.csv
├── notebooks/              # Jupyter notebooks for EDA and development
│   └── CrediTrust_RAG_Complaint_Analysis.ipynb
├── src/                    # Python source code
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Text cleaning functions
│   ├── plotting.py         # Visualization tools
│   └── ...                 
├── vector_store/           # Vector database storage
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
## Getting Started
Prerequisites
Python 3.8+

Git

Google Colab (for notebook execution)

Installation
# Clone repository
git clone https://github.com/yesufma/credi-trust-rag-chatbot.git
cd credi-trust-rag-chatbot

# Install dependencies
pip install -r requirements.txt
Usage
1. Run EDA Notebook:
jupyter notebook notebooks/CrediTrust_RAG_Complaint_Analysis.ipynb
2. Preprocess Data:
from src.data_loader import load_complaints
from src.preprocessing import clean_text, filter_products

# Load and preprocess data
df = load_complaints('data/raw_complaints.csv')
df_clean = clean_text(df)
df_filtered = filter_products(df_clean)
3. Generate Product Distribution Plot:
from src.plotting import plot_product_distribution

# Visualize complaint distribution
plt = plot_product_distribution(df_filtered)
plt.savefig('results/product_distribution.png')

🔍 Key Features
Advanced Text Processing:

🧼 Boilerplate removal ("I am writing to complain...")

📝 Lemmatization and stopword filtering

🔠 Special character and number handling

Comprehensive EDA:

📊 Product distribution analysis

📏 Narrative length statistics

🔍 Data quality assessment

RAG Pipeline:

🔎 Semantic search with FAISS/ChromaDB

💬 LLM-powered insight generation

📑 Source citation for verification

📊 Results (Task 1: EDA)
Complaint Distribution
![image](https://github.com/user-attachments/assets/fa263090-d94d-41e6-821a-77e88324adb9)
Narrative Length Analysis
![image](https://github.com/user-attachments/assets/17eccdda-5fa1-4f34-862c-1d0d3be75cd8)##

