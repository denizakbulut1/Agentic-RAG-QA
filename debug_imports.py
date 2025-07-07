# debug_imports.py (Version 2 - Catches ALL errors)

print("--- SCRIPT START ---")
# This will catch absolutely every possible error, including sys.exit()
try:
    print("1. Importing standard libraries...")
    import os
    from dotenv import load_dotenv
    print("SUCCESS: Standard libraries imported.")
    
    print("\n2. Running load_dotenv()...")
    load_dotenv()
    print("SUCCESS: load_dotenv() finished.")

    print("\n3. Importing from src.utils...")
    from src.utils import OPENAI_API_KEY
    print("SUCCESS: Imported from src.utils.")

    print("\n4. Importing from src.rag_core...")
    from src.rag_core import create_qa_chain, create_qa_chain_for_section
    print("SUCCESS: Imported from src.rag_core.")

    print("\n5. Importing from src.agent...")
    from src.agent import DocumentAgent
    print("SUCCESS: Imported from src.agent.")

    print("\n\n--- ALL IMPORTS COMPLETED SUCCESSFULLY ---")
    print("This would mean the issue is not in the Python files themselves, but a conflict with Flask.")

except BaseException as e: # Using BaseException to catch SystemExit and others
    print(f"\n!!!!!!!!!! A HARD ERROR OCCURRED !!!!!!!!!!")
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR DETAILS: {e}")
    print("\nTRACEBACK:")
    import traceback
    traceback.print_exc()