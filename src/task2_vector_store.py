import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    df = pd.read_csv('data/processed/filtered_complaints.csv')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, length_function=len
    )
    documents, metadatas = [], []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row['clean_narrative'])
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({
                "complaint_id": row.get("Complaint ID", "N/A"),
                "product": row["Product"],
            })
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(
        texts=documents, embedding=embeddings, metadatas=metadatas
    )
    os.makedirs('vector_store', exist_ok=True)
    vector_store.save_local('vector_store/faiss_index')
    print("✅ Task 2 completed: Vector store created")

if __name__ == "__main__":
    main()
