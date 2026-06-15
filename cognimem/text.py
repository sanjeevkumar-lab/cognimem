# test_import.py
try:
    from cognimem.sdk import CogniMem
    print("✅ SUCCESS: CogniMem imported correctly!")
    
    # Quick instantiation test
    mem = CogniMem(agent_id="test")
    print("✅ SUCCESS: CogniMem initialized without errors!")
except ModuleNotFoundError as e:
    print(f"❌ FAILED: {e}")
    print("Check your folder name and import statements.")