import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Any
from cogimem.models import Memory
import yaml

class EpisodicMemory:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        self.client = chromadb.PersistentClient(path=f"./chroma_db_{agent_id}")
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config["embedding"]["model"]
        )
        self.collection = self.client.get_or_create_collection(
            name="episodic_memories", 
            embedding_function=embed_fn
        )

    def add(self, memory: Memory):
        self.collection.add(
            documents=[memory.content],
            metadatas=[{
                "id": memory.id,
                "agent_id": memory.agent_id,
                "importance_score": memory.importance_score,
                "timestamp": memory.timestamp.isoformat(),
                "tags": ",".join(memory.tags)
            }],
            ids=[memory.id]
        )

    def update(self, memory: Memory):
        self.collection.update(
            ids=[memory.id],
            metadatas=[{
                "importance_score": memory.importance_score,
                "timestamp": memory.timestamp.isoformat()
            }]
        )

    def delete(self, memory_id: str):
        self.collection.delete(ids=[memory_id])

    def get_all(self) -> List[Memory]:
        results = self.collection.get()
        memories = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            memories.append(Memory(
                id=meta["id"],
                agent_id=meta["agent_id"],
                content=doc,
                memory_type="episodic",
                timestamp=datetime.fromisoformat(meta["timestamp"]),
                importance_score=float(meta["importance_score"]),
                tags=meta["tags"].split(",") if meta["tags"] else []
            ))
        return memories

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"agent_id": self.agent_id}
        )
        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            output.append({
                "content": doc,
                "id": meta["id"],
                "importance_score": float(meta["importance_score"]),
                "similarity_score": 1.0 - results["distances"][0][i] 
                
            })
        return output