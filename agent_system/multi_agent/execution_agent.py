import asyncio
from agent_system.multi_agent.graph import AgentState

class ExecutionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.max_iterations = 3

    async def run(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        iterations = 0
        history = []

        while iterations < self.max_iterations:
            # --- REASON ---
            thought = await self._think(query, history)
            history.append(f"Thought: {thought}")

            if "Final Answer" in thought:
                answer = thought.split("Final Answer:")[-1].strip()
                return {"execution_output": answer}

            # --- ACT ---
            action, code = await self._parse_action(thought)

            # --- OBSERVE ---
            observation = await self._run_code(code)
            history.append(f"Observation: {observation}")

            iterations += 1

        final = await self.llm.ainvoke(f"Summarize:\n{chr(10).join(history)}")
        return {"execution_output": final.content}

    async def _think(self, query: str, history: list) -> str:
        prompt = f"""You are a code execution agent.
            Question: {query}
            {chr(10).join(history)}

            Write code to solve this, or say "Final Answer: <result>" if done.
            Format:
            Action: execute
            Action Input: <python code>
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _parse_action(self, thought: str):
        # parse Action / Action Input from thought
        lines = thought.strip().split("\n")
        action, code = "", []
        capture = False
        for line in lines:
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                capture = True
            elif capture:
                code.append(line)
        return action, "\n".join(code)

    async def _run_code(self, code: str) -> str:
        try:
            result = await asyncio.to_thread(self._exec_sync, code)
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def _exec_sync(self, code: str) -> str:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() or "Done"