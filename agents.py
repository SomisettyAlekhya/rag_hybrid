
from retriever import HybridRetriever
from llm import generate_answer
documents = [
"Python is a high level programming language.",
"CNN stands for Convolutional Neural Network used in deep learning.",
"RAG stands for Retrieval Augmented Generation used with large language models.",
"NLP stands for Natural Language Processing which allows computers to understand and process human language.",
"LangChain is a framework for building applications with LLMs."
]
retriever = HybridRetriever(documents)

def agent_reasoning(state):
    query = state["query"].lower()

    if "error" in query or "bug" in query:
        task = "debug"
    elif "what is" in query or "explain" in query:
        task = "explain"
    else:
        task = "generate"

    state["task"] = task
    return state

def coding_agent(state):

    query = state["query"]

    context_docs = retriever.search(query)

    context = context_docs[0]

    answer = generate_answer(query, context)

    state["answer"] = answer

    return state
