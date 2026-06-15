import numpy as np
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from cognimem.core.episodic_memory import EpisodicMemory
from cognimem.core.semantic_memory import SemanticMemory
import yaml

class ConsolidationEngine:
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)["memory"]
            
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.consolidate_memories, 
            'interval', 
            hours=self.config["consolidation_interval_hours"]
        )
        self.scheduler.start()

    def ebbinghaus_decay(self, importance: float, hours_passed: float) -> float:
        k = self.config["decay_constant_k"] * (1 + importance)
        return importance * np.exp(-hours_passed / k)

    def consolidate_memories(self):
        memories = self.episodic.get_all()
        now = datetime.utcnow()
        
        for mem in memories:
            hours_passed = (now - mem.timestamp).total_seconds() / 3600
            mem.importance_score = self.ebbinghaus_decay(mem.importance_score, hours_passed)
            
            if mem.importance_score > self.config["promotion_threshold"]:
                self.semantic.add_node(mem.id, mem.content, mem.tags)
                self.episodic.update(mem)
            elif mem.importance_score < self.config["pruning_threshold"]:
                self.episodic.delete(mem.id)