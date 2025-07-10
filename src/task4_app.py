import streamlit as st
from task3_rag import RAGSystem

def main():
    st.set_page_config(page_title="CrediTrust Chatbot", layout="wide")
    st.title("CrediTrust Complaint Analyzer 🤖")
    
    if "rag" not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.rag = RAGSystem("vector_store/faiss_index")
    
    question = st.text_input("Ask about customer complaints:")
    
    if st.button("Analyze") and question:
        with st.spinner("Searching knowledge base..."):
            answer, sources = st.session_state.rag.answer_question(question)
        
        st.subheader("AI Analysis:")
        st.info(answer)
        
        with st.expander("View sources used"):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {source.page_content[:200]}...")

if __name__ == "__main__":
    main()
