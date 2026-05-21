
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from agent.prompt_templates import system_prompt
from agent.tools import TOOL_RESEARCH
from agent.rag.retriever import get_vectorstore
from agent.llms import select_llm_model

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".env")
load_dotenv(dotenv_path=env_path)
tools = TOOL_RESEARCH
vectorstore = get_vectorstore("BAAI/bge-m3")
# text-embedding-3-small
sys_msg = SystemMessage(content = system_prompt)
def build_graph(provider: str ):
    llm = select_llm_model(provider)
    llm_with_tools = llm.bind_tools(tools)
    
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        similar_question = vectorstore.similarity_search(state["messages"][0].content)

        if similar_question:
            example_msg = HumanMessage(
                content = f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}"
            )
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        else:
            # Handle the case when no similar questions are found
            return {"messages": [sys_msg] + state["messages"]}



    """
    START → Retriever → Assistant → Tools → Assistant
                     ↑              ↓
                     └──────────────┘
    """
    build_graph = StateGraph(MessagesState)
    build_graph.add_node("retriever", retriever)
    build_graph.add_node("assistant", assistant)
    build_graph.add_node("tools", ToolNode(tools))

    build_graph.add_edge(START, "retriever")
    build_graph.add_edge("retriever", "assistant")
    build_graph.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    build_graph.add_edge("tools", "assistant")
    return build_graph.compile()


# test
if __name__ == "__main__":
    question = "In the reference file, there is a column of five numbers and a column of five letters. Imagine a straight line that starts from the first number and ends at the last letter. Then imagine another straight line that starts from the last number and ends at the first letter. Now imagine another line that starts from the second number and ends at the second letter. Finally, imagine another straight line that starts from the fourth number to the fourth letter. How many intersections are formed by these imaginary lines?"
    graph = build_graph(provider="qwen")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()


