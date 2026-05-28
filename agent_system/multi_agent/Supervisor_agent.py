from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

def agent_node(state, agent, name):
    """Process state through an agent and return updated state."""
    try:
        result = agent.invoke(state)
        if not isinstance(result, dict) or "messages" not in result:
            raise ValueError(f"Agent {name} returned invalid result format: {result}")
        return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}
    except Exception as e:
        print(f"Error in agent {name}: {e}")
        return {
            "messages": [
                HumanMessage(
                    content=f"Error occurred in {name}: {str(e)}",
                    name=name
                )
            ]
        }

def create_supervisor(llm: ChatOpenAI, system_prompt: str, members: List[str]):
    """Create a supervisor agent for a team."""
    options = ["FINISH"] + members
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                "\nRespond with ONLY the name of the next role or FINISH.",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    
    def parse_output(message) -> dict:
        """Parse the output to get the next role."""
        if hasattr(message, 'content'):
            output = message.content.strip()
        else:
            output = str(message).strip()
            
        if output not in options:
            print(f"Warning: Invalid output '{output}', defaulting to FINISH")
            return {"next": "FINISH"}
        return {"next": output}
    
    chain = prompt | llm | parse_output
    return chain
