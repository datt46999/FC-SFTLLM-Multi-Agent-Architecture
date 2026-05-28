


import operator
import functools

from typing import TypedDict, List, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from lanchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END



from agent_system.multi_agent import RAG_agent, Execution_agent, External_agent, create_supervisor

def create_super_graph():
    """Create the top-level supervisor graph."""
    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    rag_chain = RAG_agent()
    Execution_chain = Execution_agent()
    External_chain = External_agent()


    supervisor_node = create_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following teams: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["ResearchTeam", "PaperWritingTeam"],
    )
    
    def get_last_message(state: State) -> dict:
        """Get the last message from state and return it as a dictionary."""
        return {"messages": [state["messages"][-1]]}
    
    def join_graph(response: dict) -> dict:
        """Join the graph response with the current state."""
        return {"messages": [response["messages"][-1]]}

    super_graph = StateGraph(State)
    super_graph.add_node("RAGTeam", get_last_message | rag_chain | join_graph)
    super_graph.add_node("ExecutionTeam", get_last_message | Execution_chain| join_graph)
    super_graph.add_node("ExternalTeam", get_last_message | External_chain | join_graph)
    super_graph.add_node("supervisor", supervisor_node)

    super_graph.add_edge("RAGTeam", "supervisor")
    super_graph.add_edge("ExecutionTeam", "supervisor")
    super_graph.add_edge("ExternalTeam", "ExternalTeam")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "RAGTeam": "RAGTeam",
            "ExecutionTeam": "ExecutionTeam",
            "ExternalTeam": "ExternalTeam",
            "FINISH": END,
        },
    )
    super_graph.add_edge(START, "supervisor")
    
    return super_graph.compile()



def main():
    """Run the hierarchical agent system with example queries."""
    super_graph = create_super_graph()

    print("Example: Research Report on North American Sturgeon")
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write a brief research report on the North American sturgeon. Include a chart."
            )
        ],
        "next": "ResearchTeam"
    }
    
    for s in super_graph.stream(
        initial_state,
        {"recursion_limit": 150},
    ):
        if "__end__" not in s:
            print(s)
            print("----")

if __name__ == "__main__":
    main() 