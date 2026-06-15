from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from sdk import CogniMem

class CogniMemChatHistory(BaseChatMessageHistory):
    def __init__(self, agent_id: str):
        self.memory = CogniMem(agent_id=agent_id)

    @property
    def messages(self):
        # For LangChain, we return recent working memory as messages
        wm = self.memory.wm.get_all()
        messages = []
        for m in wm:
            if "user" in m.content.lower():
                messages.append(HumanMessage(content=m.content))
            else:
                messages.append(AIMessage(content=m.content))
        return messages

    def add_message(self, message: BaseMessage):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        importance = 0.8 if isinstance(message, HumanMessage) else 0.5
        self.memory.remember(f"{role}: {message.content}", importance=importance, tags=[role])

    def clear(self):
        self.memory.wm.clear()