import operator
import functools

from typing import TypedDict, List, Annotated

from langchain_core.messages import BaseMessage
from lanchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

from agent_system.multi_agent.Supervisor_agent import agent_node, create_supervisor
from agent_system.tools import IMAGE_GENERATE_TOOLS, CODE_INTERPRETER_TOOLS, DOCUMENT_PROCESSING_TOOLS, MATHEMATICAL_TOOLS


def Execution_agent():
    """  Create external agent search """
    class ExcutionTeamState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
       
        next: str

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    image_agent = create_react_agent(llm, tools=IMAGE_GENERATE_TOOLS)
    imge_node = functools.partial(agent_node, agent=image_agent, name="ImageGenerate")

    code_interp_agent = create_react_agent(llm, tools=CODE_INTERPRETER_TOOLS)
    code_interp_node = functools.partial(agent_node, agent=code_interp_agent, name="CodeInterpreter")

    docs_agent = create_react_agent(llm, tools=DOCUMENT_PROCESSING_TOOLS,)
    docs_node = functools.partial(agent_node, agent=docs_agent, name="Document")

    mathem_agent = create_react_agent(llm, tools=MATHEMATICAL_TOOLS)
    mathem_node = functools.partial(agent_node, agent=mathem_agent, name="Mathenatical")
    
    supervisor_agent = create_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  ExternalTool. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["ExternalTool"],
    )

    execute_graph = StateGraph(ExcutionTeamState)

    execute_graph.add_node("ImageGenerate", imge_node)
    execute_graph.add_node("CodeInterpreter", code_interp_node )
    execute_graph.add_node("Document", docs_node)
    execute_graph.add_node("Mathenatical", mathem_node)

 
    execute_graph.add_edge("ImageGenerate", "supervisor")
    execute_graph.add_edge("CodeInterpreter", "supervisor")
    execute_graph.add_edge("Document", "supervisor")
    execute_graph.add_edge("Mathenatical", "supervisor")
    execute_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"ImageGenerate": "ImageGenerate",
         "CodeInterpreter": "CodeInterpreter",
         "Document": "Document",
         "Mathenatical": "Mathenatical",
         "FINISH": END},
    )
    execute_graph.add_edge(START, "supervisor")
    
    return execute_graph.compile()
