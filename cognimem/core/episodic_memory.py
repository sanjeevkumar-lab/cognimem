import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Any
from cognimem.models import Memory
import yaml

class EpisodicMemory:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print("⚠️ Warning: config.yaml not found in root directory. Using default settings.")
            config = {}
        
        embedding_model = config.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path=f"./chroma_db_{agent_id}")
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="episodic_memories", 
            embedding_function=embed_fn  # type: ignore
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
        documents: List = results.get("documents") or []
        metadatas: List = results.get("metadatas") or []
        for i, doc in enumerate(documents):
            meta: Dict = metadatas[i] if i < len(metadatas) else {}
            try:
                importance_value = meta.get("importance_score", 0.0)
                if isinstance(importance_value, (int, float)):
                    importance_score = float(importance_value)
                else:
                    importance_score = 0.0
            except (ValueError, TypeError):
                importance_score = 0.0
            memories.append(Memory(
                id=str(meta.get("id", "")),
                agent_id=str(meta.get("agent_id", "")),
                content=doc,
                memory_type="episodic",
                timestamp=datetime.fromisoformat(str(meta.get("timestamp", datetime.now().isoformat()))),
                importance_score=importance_score,
                tags=str(meta.get("tags", "")).split(",") if meta.get("tags") else []
            ))
        return memories

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"agent_id": self.agent_id}
        )
        output = []
        documents: List = results.get("documents") or []
        metadatas: List = results.get("metadatas") or []
        distances: List = results.get("distances") or []
        
        docs_list: List = documents[0] if documents and len(documents) > 0 else []
        metas_list: List = metadatas[0] if metadatas and len(metadatas) > 0 else []
        dists_list: List = distances[0] if distances and len(distances) > 0 else []
        
        for i, doc in enumerate(docs_list):
            meta: Dict = metas_list[i] if i < len(metas_list) else {}
            try:
                importance_value = meta.get("importance_score", 0.0)
                if isinstance(importance_value, (int, float)):
                    importance_score = float(importance_value)
                else:
                    importance_score = 0.0
            except (ValueError, TypeError):
                importance_score = 0.0
            distance = dists_list[i] if i < len(dists_list) else 0.0
            output.append({
                "content": doc,
                "id": str(meta.get("id", "")),
                "importance_score": importance_score,
                "similarity_score": 1.0 - distance
            })
        return output