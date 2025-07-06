# Final version of src/main.py

import argparse
from src.agent import setup_agent_executor
from langchain_core.messages import HumanMessage, AIMessage
import os

def main():
    """
    Main function to run the agent in an efficient, stateful, conversational loop.
    This version includes a caching mechanism for RAG pipelines.
    """
    parser = argparse.ArgumentParser(description="Query a PDF document interactively.")
    parser.add_argument("file_path", type=str, help="The path to the PDF file.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at '{args.file_path}'")
        return

    # --- KEY CHANGE 1: Introduce the RAG Chain Cache ---
    # This dictionary will store our expensive-to-create RAG chains.
    # The keys will be unique identifiers (e.g., the file path for the full doc,
    # or 'filepath_startpage_endpage' for sections).
    rag_chain_cache = {}
    print("Initialized RAG chain cache for this session.")

    print("Setting up agent... This may take a moment.")
    try:
        # --- KEY CHANGE 2: Pass the Cache to the Agent Setup Function ---
        # The agent setup function now needs access to the cache so that its
        # tools can check for and store RAG chains.
        agent_executor = setup_agent_executor(args.file_path, rag_chain_cache)
    except Exception as e:
        print(f"Failed to set up the agent. Error: {e}")
        return

    print("\n--- Document Analysis Agent ---")
    print(f"Setup complete. Talking about document: {os.path.basename(args.file_path)}")
    print("You can now ask questions. Type 'quit' or 'exit' to end the conversation.")

    # Initialize the chat history list
    chat_history = []

    while True:
        try:
            query = input("\nYour Question: ")
            
            if query.lower() in ["quit", "exit"]:
                print("Exiting. Goodbye!")
                break

            if not query:
                continue

            # --- ADD THIS BLOCK FOR DEBUGGING ---
            print("\n==================== DEBUG: HISTORY SENT TO AGENT ====================")
            if not chat_history:
                print("[]  (History is empty)")
            else:
                for msg in chat_history:
                    print(msg)
            print("====================================================================\n")
            # --- END OF DEBUG BLOCK ---

            print("Agent is thinking...")
            
            result = agent_executor.invoke({
                "input": query,
                "chat_history": chat_history,
                "file_path": args.file_path 
            })

            final_answer = result["output"]
            print("\nAssistant:")
            print(final_answer)

            # Update the history with the latest turn
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=final_answer))

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during conversation: {e}")
            # For debugging, you might want more detail:
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()