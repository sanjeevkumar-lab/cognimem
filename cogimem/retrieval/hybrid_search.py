from sentence_transformers import SentenceTransformer
from cogimem.core.episodic_memory import EpisodicMemory
from cogimem.core.semantic_memory import SemanticMemory
from cogimem.models import Memory
import yaml

class HybridRetriever:
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)["retrieval"]
        self.model = SentenceTransformer(self.config.get("embedding_model_override", "all-MiniLM-L6-v2"))
        self.episodic = episodic
        self.semantic = semantic

    def retrieve(self, query: str, agent_id: str, top_k: int = 5) -> list:
        query_embedding = self.model.encode(query).tolist()
        
        episodic_results = self.episodic.similarity_search(query_embedding, top_k=top_k * 2)
        semantic_results = self.semantic.get_related_facts(query, top_k=top_k)
        
        scored_results = []
        for res in episodic_results + semantic_results:
            time_decay_factor = res["importance_score"]
            final_score = (res["similarity_score"] * self.config["vector_weight"]) + \
                          (time_decay_factor * self.config["importance_weight"])
            
            if not any(r["id"] == res["id"] for r in scored_results):
                scored_results.append({**res, "final_score": final_score})
                
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        final_memories = []
        for item in scored_results[:top_k]:
            final_memories.append(Memory(
                id=item["id"],
                agent_id=agent_id,
                content=item["content"],
                memory_type="hybrid",
                importance_score=item["importance_score"]
            ))
        return final_memories