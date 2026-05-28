import os
import operator
import functools
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated

from langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import crete_react_agent 


from agent_system.rag.retriever import top_rags
from agent_system.multi_agent.Supervisor_agent import create_supervisor, agent_node


PATH_VECTORSTORE = ""
NUM_RETRIEVED_DOCS = 30
NUM_DOCS_FINAL = 5



_embeddings = OpenAIEmbeddings()
_knowledge_index: FAISS = FAISS.load_local(
    PATH_VECTORSTORE, _embeddings, allow_dangerous_deserialization=True
)
_reranker = None 

@tool
def retriever_from_docs(question: str) -> str:
    """
    Search the document knowledge base and return the most relevant passages
    for the given question. Use this tool whenever you need factual information
    from the documents.

    Args:
        question: The question or query to look up in the knowledge base.

    Returns:
        A numbered list of the most relevant document passages.
    """
    relevant_docs = top_rags(
        question=question,
        knowledge_index=_knowledge_index,
        reranker=_reranker,
        num_retrieved_docs=NUM_RETRIEVED_DOCS,
        num_docs_final=NUM_DOCS_FINAL,
    )

    if not relevant_docs:
        return "No relevant documents found for the given question."

    # Format as a readable string so the LLM can consume it easily
    formatted = "\n\n".join(
        f"[{i+1}] {doc}" for i, doc in enumerate(relevant_docs)
    )
    return formatted

def RAG_agent():
    """ Create RAG agent for retriever documnent"""
    class RAG_state(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str 
    llm = ChatOpenAI(model = "gpt-4", temperature = 0)

    retriever_agent = crete_react_agent(llm, tools =[retriever_from_docs])
    retriever_node = functools.partial(agent_node, agent = retriever_agent, name = "RAG_agent")
    
    supervisor_agent = create_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  retriever_from_docs . Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["RAG_agent"],
    )

    rag_graph = StateGraph(RAG_state)
    rag_graph.add_node("RAG_agent", retriever_agent)
    rag_graph.add_node("supervisor", supervisor_agent)
    
    rag_graph.add_edge("RAG_agent", "supervisor")
    rag_graph.add_conditional(
        "supervisor",
        lambda x: x["next"],
        {"RAG_agent": "RAG_agent", "FINISH": END}
    )
    rag_graph.add_edge(START, "supervisor")
    return rag_graph.compile()
