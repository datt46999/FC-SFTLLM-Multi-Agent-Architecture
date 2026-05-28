import operator
import functools

from typing import TypedDict, List, Annotated

from langchain_core.messages import BaseMessage
from lanchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

from agent_system.multi_agent.Supervisor_agent import agent_node, create_supervisor
from agent_system.tools import EXTERNAL_TOOLs


def External_agent():
    """  Create external agent search """
    class ResearchTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
       
        next: str

    llm = ChatOpenAI(model="gpt-4", temperature=0)

   
    research_agent = create_react_agent(llm, tools=EXTERNAL_TOOLs)
    research_node = functools.partial(agent_node, agent=research_agent, name="ExternalTool")
    
    supervisor_agent = create_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  ExternalTool. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["ExternalTool"],
    )

    research_graph = StateGraph(ResearchTeamState)

    research_graph.add_node("ExternalTool", research_node)
    research_graph.add_node("supervisor", supervisor_agent)

 
    research_graph.add_edge("ExternalTool", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"ExternalTool": "ExternalTool", "FINISH": END},
    )
    research_graph.add_edge(START, "supervisor")
    
    return research_graph.compile()
