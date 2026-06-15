from collections import deque
from typing import List
from cognimem.models import Memory

class WorkingMemory:
    def __init__(self, limit: int = 10):
        self.limit = limit
        self.buffer = deque(maxlen=limit)

    def add(self, memory: Memory):
        memory.memory_type = "working"
        self.buffer.append(memory)

    def get_all(self) -> List[Memory]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()