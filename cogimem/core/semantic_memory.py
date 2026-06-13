import json
import networkx as nx
from typing import List, Dict

class SemanticMemory:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.graph = nx.Graph()
        self.graph_file = f"./semantic_graph_{agent_id}.json"
        self._load()

    def _load(self):
        try:
            data = json.load(open(self.graph_file))
            self.graph = nx.node_link_graph(data)
        except FileNotFoundError:
            pass

    def _save(self):
        data = nx.node_link_data(self.graph)
        with open(self.graph_file, "w") as f:
            json.dump(data, f)

    def add_node(self, memory_id: str, content: str, tags: List[str]):
        self.graph.add_node(memory_id, content=content, tags=tags, type="fact")
        for tag in tags:
            self.graph.add_node(tag, type="concept")
            self.graph.add_edge(memory_id, tag, weight=1.0)
        self._save()

    def get_related_facts(self, query: str, top_k: int = 5) -> List[Dict]:
        query_lower = query.lower()
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "fact" and any(tag.lower() in query_lower for tag in data.get("tags", [])):
                results.append({
                    "id": node,
                    "content": data["content"],
                    "importance_score": 1.0, 
                    "similarity_score": 0.8 
                })
        return results[:top_k]