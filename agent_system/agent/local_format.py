import re
import json 

from typing import List, Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

def _tools_to_xlam_schema(tools: list[BaseTool]) -> str:
    """Convert LangChain tools to xLAM <tools> XML schema format."""
    tools_list = []
    for tool in tools:
        # pull parameter schema from tool
        schema = tool.args_schema.schema() if tool.args_schema else {"properties": {}}
        parameters = {
            k: {"type": v.get("type", "string"), "description": v.get("description", "")}
            for k, v in schema.get("properties", {}).items()
        }
        tools_list.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": parameters}
        })
    return json.dumps(tools_list, indent=2)

def _build_xlam_prompt(query: str, tools: list[BaseTool]) -> str:
    """Build the full xLAM-formatted prompt."""
    tools_schema = _tools_to_xlam_schema(tools)
    return (
        f"<user>{query}</user>\n\n"
        f"<tools>{tools_schema}</tools>"
    )

def _parse_xlam_calls(response_text: str) -> list[dict]:
    """Extract tool calls from xLAM <calls> XML block."""
    match = re.search(r"<calls>(.*?)</calls>", response_text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return []
    

def _execute_tool_calls(
    tool_calls: list[dict],
    tools: list[BaseTool]
) -> str:
    """Execute parsed tool calls and return combined results."""
    tool_map = {tool.name: tool for tool in tools}
    results = []
    for call in tool_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        if name in tool_map:
            print(f"[xLAM] Calling tool: {name} with args: {args}")
            result = tool_map[name].invoke(args)
            results.append(f"<tool_result name='{name}'>\n{result}\n</tool_result>")
        else:
            results.append(f"<tool_result name='{name}'>Tool not found.</tool_result>")
    return "\n\n".join(results)


class Format_agent:
    def __init__(self, llm, tools: List[BaseTool], max_iterations: int)-> str:
        self.llm = llm
        self.tools  = tools
        self.max_iterations = max_iterations

    def invoke(self,state: dict)->dict:
        """
        Replacement for create_react_agent's invoke
        """
        messages = state.get("messages", [])
        query = ''
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break 
        
        conversation = []
        prompt = _build_xlam_prompt(query, self.tools)


        for i in range(self.max_iterations):
            print(f"Interaction{i+1}")

            full_input = prompt
            if conversation:
                full_input += "\n\n" + "\n\n".join(conversation)

            response = self.llm.invoke(full_input)
            response_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            conversation.append(f"<assistant>{response_text}</assistant>")

            tool_calls = _parse_xlam_calls(response_text)

            if not tool_calls:

                print("[xLAM] No tool calls — returning final answer.")
                return {
                    "messages": messages + [AIMessage(content=response_text)]
                }

  
            tool_results = _execute_tool_calls(tool_calls, self.tools)
            conversation.append(f"<tool_results>{tool_results}</tool_results>")
            print(f"[xLAM] Tool results fed back, continuing...")


        return {
            "messages": messages + [AIMessage(content=response_text)]
        }