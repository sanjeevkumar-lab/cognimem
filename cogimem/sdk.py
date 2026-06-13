from cogimem.core.working_memory import WorkingMemory
from cogimem.core.episodic_memory import EpisodicMemory
from cogimem.core.semantic_memory import SemanticMemory
from cogimem.core.consolidation import ConsolidationEngine
from cogimem.retrieval.hybrid_search import HybridRetriever
from cogimem.models import Memory
import yaml

class CogniMem:
    def __init__(self, agent_id: str = "default_agent"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        self.agent_id = agent_id
        self.wm = WorkingMemory(limit=config["memory"]["working_memory_limit"])
        self.em = EpisodicMemory(agent_id)
        self.sm = SemanticMemory(agent_id)
        self.consolidator = ConsolidationEngine(self.em, self.sm)
        self.retriever = HybridRetriever(self.em, self.sm)

    def remember(self, content: str, importance: float = 0.5, tags: list = None):
        tags = tags or []
        mem = Memory(
            agent_id=self.agent_id, 
            content=content, 
            memory_type="episodic", 
            importance_score=importance, 
            tags=tags
        )
        self.wm.add(mem)
        self.em.add(mem)
        return mem.id

    def recall(self, query: str, top_k: int = 3) -> str:
        memories = self.retriever.retrieve(query, self.agent_id, top_k)
        if not memories:
            return "No relevant memories found."
        return "\n".join([
            f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] (Relevance: {m.importance_score:.2f}): {m.content}" 
            for m in memories
        ])