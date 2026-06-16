import operator
import functools
import re
from flashrank import Ranker, RerankRequest
from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser

from agent_system.tools import TOOLS_RESEARCHER, TOOLS_SCRAPE, CODE_INTERPRETER
from agent_system.rag.retriever import top_rags
from agent_system.agent.system_prompts import system_prompt, researcher_prompt, rag_prompt, scraper_prompt, coder_prompt
from agent_system.agent.get_model import get_llm
# from agent_system.agent.local_loader import Format_agent
from langchain_huggingface import HuggingFaceEmbeddings



from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

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
    allow_dangerous_deserialization=True 
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

members = ["RE_Retriever","Researcher", "ScraperWeb", "Coder"]
options = ["FINISH"] + members


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name = "messages"),
        (
            "system",
            "Give the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options =str(options), members = ", ".join(members))

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


_ROUTING_SUFFIX = (
               "Give the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
)

SUPERVISOR_PROMPT_LOCAL = ChatPromptTemplate.from_messages([
    ("system", system_prompt ),  
    MessagesPlaceholder(variable_name="messages"),
    ("human",_ROUTING_SUFFIX),    
]).partial(options=str(options), members=", ".join(members))

_VALID = set(options)  # {"FINISH", "RE_Retriever", "Researcher", "ScraperWeb", "Coder"}

def _parse_next(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    print(f"[DEBUG] parsed assistant output: '{text}'")
    

    for token in _VALID:
        if token.lower() == text.lower():
            return token

    for token in _VALID:
        if token.lower() in text.lower() or text.lower() in token.lower():
            return token
    
    return "FINISH"
def _make_agent(llm, tools, backend, name):
    if backend == "LOCAL":
        return create_agent(llm, tools = tools)
    else:
        return create_react_agent(llm, tools =tools)
       
    
def agent_node(state, agent, name):
    """
    Process state through  an agent and return update state
    """
    results = agent.invoke(state)
    return {"messages": [HumanMessage(content = results["messages"][-1].content, name = name)]}
    
def supervisor_agent(state):
    if backend == "OPENAI":
        llm,_ = get_llm()
        suppervisor_chain = prompt | llm.with_structured_output(RouterResponse) 
        return  suppervisor_chain.invoke(state, config={"callbacks": [langfuse_handler]})
    
    else:
        llm, _ = get_llm()
        chain = SUPERVISOR_PROMPT_LOCAL | llm | StrOutputParser()

        raw_text = chain.invoke(state, config={"callbacks": [langfuse_handler]})
        print(f"[DEBUG] raw_text: {repr(raw_text)}") 
        decision = _parse_next(raw_text)
        
        # Return a dictionary with the update
        return {"next": decision}
        


      
def create_graph():
    """ 
    Create configure the supervisor graph
    """
    llm, _ = get_llm()

    RAG_agent = _make_agent(llm, [retriever_tool], backend,name = "RE_Retriever")
    RAG_node = functools.partial(agent_node, agent = RAG_agent, name = "RE_Retriever")

    researcher_agent = _make_agent(llm, researcher_tools, backend, name = "Researcher")
    researcher_node = functools.partial(agent_node, agent = researcher_agent, name = "Researcher")

    scrape_agent = _make_agent(llm, scrape_tools, backend, name = "ScraperWeb")
    scrape_node = functools.partial(agent_node, agent = scrape_agent, name = "ScraperWeb")

    Coder_agent = _make_agent(llm, [ python_repl_tool, code_interpreter], backend, name = "Coder")
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


def test(query):
    """
    Run supervisor system with example queries
    """
    graph = create_graph()



    for s in graph.stream(
        {
            "messages":[
                HumanMessage(content=query)
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            print("---"*100)

    
    # result = graph.invoke(
    # {
    #     "messages": [HumanMessage(content=query)]
    # }
    # )
    # print("---"*100)
    # print(result["messages"][-1].content)
           

if __name__ =="__main__":
    while True:
        query = input("Your query: ")
        if query in ['EXIT', 'Exit', 'exit']:
            print("==="*100)
            print("SYSTEM: CLOSE")
            print("==="*100)
            break 
        else:
            
            test(query)





    
