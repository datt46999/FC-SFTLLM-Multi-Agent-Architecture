from agent_system.multi_agent.graph import AgentState

class ExternalAgent:
    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search_tool = search_tool
        self.max_iterations = 3

    async def run(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        history = []
        iterations = 0

        while iterations < self.max_iterations:
            # --- REASON ---
            thought = await self._think(query, history)
            history.append(f"Thought: {thought}")

            if "Final Answer" in thought:
                answer = thought.split("Final Answer:")[-1].strip()
                return {"external_output": answer}

            # --- ACT ---
            _, search_query = await self._parse_action(thought)

            # --- OBSERVE ---
            results = await self.search_tool.arun(search_query)
            history.append(f"Observation: {results}")

            iterations += 1

        final = await self.llm.ainvoke(f"Summarize:\n{chr(10).join(history)}")
        return {"external_output": final.content}

    async def _think(self, query, history):
        prompt = f"""You are a web search agent.
        Question: {query}
        {chr(10).join(history)}

        Search for info or say "Final Answer: <answer>" if done.
        Format:
        Action: search
        Action Input: <search query>
        """
        res = await self.llm.ainvoke(prompt)
        return res.content

    async def _parse_action(self, thought: str):
        lines = thought.strip().split("\n")
        action, action_input = "", ""
        for line in lines:
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            if line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()
        return action, action_input