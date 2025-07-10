import pandas as pd
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

class RAGSystem:
    def __init__(self, vector_store_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = FAISS.load_local(vector_store_path, self.embeddings)
        self.generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base"
        )
    
    def retrieve(self, query, top_k=5):
        return self.db.similarity_search(query, k=top_k)
    
    def generate_answer(self, context_chunks, question):
        context_text = "\n\n".join([doc.page_content for doc in context_chunks])
        prompt = f"""You are a financial analyst. Use this context:
        {context_text}
        Question: {question}
        Answer:"""
        return self.generator(prompt, max_length=256)[0]['generated_text']
    
    def answer_question(self, question):
        context_chunks = self.retrieve(question)
        return self.generate_answer(context_chunks, question), context_chunks

def main():
    rag = RAGSystem("vector_store/faiss_index")
    questions = [
        "What are common credit card issues?",
        "How do I dispute a BNPL charge?",
        "Why was my loan application denied?"
    ]
    results = []
    for q in questions:
        answer, sources = rag.answer_question(q)
        results.append({
            "Question": q, 
            "Answer": answer,
            "Sources": [s.page_content[:100] + "..." for s in sources]
        })
    pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
    print("✅ Task 3 completed: RAG evaluation done")

if __name__ == "__main__":
    main()
