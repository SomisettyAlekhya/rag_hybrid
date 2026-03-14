
import streamlit as st
from graph import create_graph

graph = create_graph()

st.set_page_config(page_title="Hybrid RAG Assistant")

st.title("Hybrid RAG • LangGraph • AI Coding Assistant")

query = st.text_input("Ask a coding question")

if query:
    result = graph.invoke({"query": query})

    st.write("### You")
    st.write(query)

    st.write("### AI Assistant")
    st.write(result["answer"])
