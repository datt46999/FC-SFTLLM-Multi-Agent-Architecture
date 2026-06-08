import operator
import functools
from flashrank import Ranker, RerankRequest
from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_community.vectorstores import FAISS


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from agent_system.tools import TOOLS_RESEARCHER, TOOLS_SCRAPE, CODE_INTERPRETER
from agent_system.rag.retriever import top_rags
from agent_system.agent.system_prompts import system_prompt
from agent_system.agent.model_Finetune import get_llm
from agent_system.agent.local_format import Format_agent
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # must match what you used in embedding.py

path_knowledge_index = "./data/vectorstore/faiss_index"

load_dotenv()


RERANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

"""
===================================================
Tools collection
===================================================
"""

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)


path_knowledge_index = "./data/vectorstore/faiss_index"  # set your FAISS index path here
knowledge_index: FAISS =FAISS.load_local(
    path_knowledge_index,
    embedding_model,
    allow_dangerous_deserialization=True   # required for loading .pkl files
)
@tool
def retriever_tool(query: str) -> str:
    """Search the local knowledge base (RAG) for relevant documents.

    Args:
        query: The search query."""
    docs = top_rags(
        question=query,
        knowledge_index=knowledge_index,
        reranker=RERANKER,
        num_retrieved_docs=30,
        num_docs_final=5,
    )
    return "\n\n---\n\n".join(
        [f"<Document>\n{doc}\n</Document>" for doc in docs]
    )


researcher_tools = TOOLS_RESEARCHER
scrape_tools = TOOLS_SCRAPE


code_interpreter = CODE_INTERPRETER
python_repl_tool = PythonREPLTool()


"""
===================================================
system
===================================================
"""
prompt = system_prompt
members = ["RE_Retriever","Researcher", "ScraperWeb", "Coder"]
options = ["FINISH"] + members

"""
===================================================
graph
===================================================
"""
llm, backend = get_llm()
class RouterResponse(BaseModel):
    next:  Literal["FINISH", "RE_Retriever","Researcher", "ScraperWeb","Coder"]

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    next: str



def _make_agent(llm, tools, backend, max_iterations: int = 2):
    if backend == "LOCAL":
        return Format_agent(llm, tools, max_iterations=max_iterations)
    else:
        return create_react_agent(llm, tools)
def agent_node(state, agent, name):
    """
    Process state through  an agent and return update state
    """
    results = agent.invoke(state)
    return {"messages": [HumanMessage(content = results["messages"][-1].content, name = name)]}
    

def supervisor_agent(state):
    if backend in ("OPENAI", "OPENROUTER"):
        supervisor_chain = prompt | llm.with_structured_output(RouterResponse)
        result = supervisor_chain.invoke({
            "messages": state["messages"],
            "members": ", ".join(members),
            "options": ", ".join(options),
        })
        print(f"[Supervisor] routing to: {result.next}")
        return {"next": result.next} 
    else:  # LOCAL
       
        supervisor_chain = prompt | llm
        response = supervisor_chain.invoke({
            "messages": state["messages"],
            "members": ", ".join(members),
            "options": ", ".join(options),
        })
        text = response.content if hasattr(response, "content") else str(response)
        print(f"[Supervisor LOCAL] raw: {text}")
        for option in options:
            if option in text:
                print(f"[Supervisor LOCAL] routing to: {option}")
                return {"next": option}
        return {"next": "FINISH"}
        # fallback
      
def create_graph():
    """ 
    Create configure the supervisor graph
    """


    RAG_agent = _make_agent(llm, [retriever_tool], backend)
    RAG_node = functools.partial(agent_node, agent = RAG_agent, name = "RE_Retriever")

    researcher_agent = _make_agent(llm, researcher_tools, backend)
    researcher_node = functools.partial(agent_node, agent = researcher_agent, name = "Researcher")

    scrape_agent = _make_agent(llm, scrape_tools, backend)
    scrape_node = functools.partial(agent_node, agent = scrape_agent, name = "ScraperWeb")

    Coder_agent = _make_agent(llm, [ python_repl_tool, code_interpreter], backend)
    Coder_node = functools.partial(agent_node, agent = Coder_agent , name = "Coder")

    workflow = StateGraph(AgentState)

    workflow.add_node("RE_Retriever", RAG_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("ScraperWeb", scrape_node)
    workflow.add_node("Coder", Coder_node)
    workflow.add_node("supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "supervisor")

    conditional_map ={k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_edge(START, "supervisor")
    return workflow.compile()


def test():
    """
    Run supervisor system with example queries
    """
    graph = create_graph()



    for s in graph.stream(
        {
            "messages":[
                HumanMessage(content="What's the weather in Ho Chi Minh today?")
            ]
        },
        config={"recursion_limit": 20} 

    ):
        if "__end__" not in s:
            print(s)
            print("---"*100)


if __name__ =="__main__":
    test()

    



    
