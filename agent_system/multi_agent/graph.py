from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    user_query: str
    
    # ReAct loop fields
    thought: Optional[str]        # agent's reasoning
    action: Optional[str]         # what tool to call
    action_input: Optional[str]   # tool input
    observation: Optional[str]    # tool result
    iterations: int               # prevent infinite loop
    
    # Outputs
    rag_output: Optional[str]
    execution_output: Optional[str]
    external_output: Optional[str]
    final_answer: Optional[str]
    
    messages: Annotated[list, operator.add]
    next: str

