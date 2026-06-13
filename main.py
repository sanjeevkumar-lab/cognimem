import time
from cogimem.sdk import CogniMem

def main():
    print("Initializing CogniMem Agent...")
    agent = CogniMem(agent_id="test_agent_01")

    print("\n--- 1. Storing Memories (Episodic) ---")
    agent.remember("The user's name is Alice and she loves Python programming.", importance=0.9, tags=["user_profile", "python"])
    agent.remember("We discussed deploying the app using Docker yesterday.", importance=0.6, tags=["project", "docker"])
    agent.remember("The weather was rainy today.", importance=0.2, tags=["weather"])

    print("\n--- 2. Immediate Recall (Hybrid Search) ---")
    query = "What does the user like to code in?"
    print(f"Query: '{query}'")
    print("Recall Result:\n", agent.recall(query, top_k=2))

    print("\n--- 3. Simulating Time Passing (Decay) ---")
    print("Manually triggering consolidation to simulate 48 hours passing...")
    from datetime import datetime, timedelta
    from cogimem.core.consolidation import ConsolidationEngine
    

    memories = agent.em.get_all()
    for mem in memories:
        mem.timestamp = datetime.utcnow() - timedelta(hours=48)
        k = 24 * (1 + mem.importance_score)
        hours_passed = 48
        mem.importance_score = mem.importance_score * (2.71828 ** (-hours_passed / k))
        agent.em.update(mem)
        print(f"  - '{mem.content[:30]}...' decayed to score: {mem.importance_score:.2f}")

    print("\n--- 4. Recall After Decay ---")
    print("Query: 'What was the weather?'")
    print("Result:", agent.recall("weather", top_k=1)) 
    
    print("Query: 'What is the user's name and coding preference?'")
    print("Result:\n", agent.recall("user name coding", top_k=1)) 

if __name__ == "__main__":
    main()