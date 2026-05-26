import asyncio
from agent_system.multi_agent.graph import AgentState

from agent_system.multi_agent import *



class MergerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        parts = []
        
        if state.get("rag_output"):
            parts.append(f"[Knowledge Base]\n{state['rag_output']}")
        if state.get("execution_output"):
            parts.append(f"[Execution Result]\n{state['execution_output']}")
        if state.get("external_output"):
            parts.append(f"[External Search]\n{state['external_output']}")
        
        combined = "\n\n".join(parts)
        
        final = self.llm.invoke(
            f"Synthesize these results into one clear answer:\n\n"
            f"{combined}\n\nQuestion: {state['user_query']}"
        )
        
        return {"final_answer": final.content, "next": "END"}
    

class Orchestrator:
    def __init__(self, llm, vectorstore, search_tool):
        self.rag      = RAGAgent(vectorstore, llm)
        self.execution = ExecutionAgent(llm)
        self.external  = ExternalAgent(llm, search_tool)
        self.merger    = MergerAgent(llm)

    async def run(self, query: str) -> str:
        state = AgentState(user_query=query, messages=[], next="", agents_needed=[])

        rag_result, exec_result, ext_result = await asyncio.gather(
            self.rag.run(state),
            self.execution.run(state),
            self.external.run(state),
        )

        # Merge all outputs
        merged_state = {
            **state,
            **rag_result,
            **exec_result,
            **ext_result,
        }

        final = await self.merger.run(merged_state)
        return final["final_answer"]

# Entry point
async def main():
    orchestrator = Orchestrator(llm=..., vectorstore=..., search_tool=...)
    answer = await orchestrator.run("Explain async RAG agents")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())