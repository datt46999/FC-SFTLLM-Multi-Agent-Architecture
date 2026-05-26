import asyncio
from typing import Optional
from langchain_community.vectorstores import FAISS
from agent_system.multi_agent.graph import AgentState
from agent_system.rag.retriever import top_rags

class RAGAgent:
    def __init__(
        self,
        llm,
        knowledge_index: FAISS,
        reranker=None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ):
        self.llm = llm
        self.knowledge_index = knowledge_index
        self.reranker = reranker
        self.num_retrieved_docs = num_retrieved_docs
        self.num_docs_final = num_docs_final
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
                return {"rag_output": answer}

            # --- ACT: use top_rags to retrieve ---
            search_query = await self._parse_search_query(thought)

            # --- OBSERVE ---
            observation = await self._retrieve(search_query)
            history.append(f"Observation (top docs):\n{observation}")

            iterations += 1

        # Fallback
        final = await self._finalize(query, history)
        return {"rag_output": final}

    async def _think(self, query: str, history: list) -> str:
        history_str = "\n".join(history)
        prompt = f"""You are a RAG agent. Reason step by step.

        Question: {query}
        {history_str}

        If you have enough info say: "Final Answer: <answer>"
        If you need to search say:
        Action: retrieve
        Action Input: <your search query>
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _parse_search_query(self, thought: str) -> str:
        for line in thought.strip().split("\n"):
            if line.startswith("Action Input:"):
                return line.replace("Action Input:", "").strip()
        return ""

    async def _retrieve(self, search_query: str) -> str:
        """Run top_rags in thread so it doesn't block async loop."""
        docs = await asyncio.to_thread(
            top_rags,
            question=search_query,
            knowledge_index=self.knowledge_index,
            reranker=self.reranker,
            num_retrieved_docs=self.num_retrieved_docs,
            num_docs_final=self.num_docs_final,
        )

        # Format docs for the observation
        formatted = "\n\n".join(
            [f"[Doc {i+1}]: {doc}" for i, doc in enumerate(docs)]
        )
        return formatted

    async def _finalize(self, query: str, history: list) -> str:
        history_str = "\n".join(history)
        response = await self.llm.ainvoke(
            f"Based on research:\n{history_str}\n\nProvide final answer for: {query}"
        )
        return response.content