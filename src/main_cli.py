# src/main.py

import argparse
import os
from src.agent import DocumentAgent
from langchain_core.messages import HumanMessage, AIMessage

def main():
    """
    Main function to run the agent in an efficient, stateful, conversational loop.
    """
    parser = argparse.ArgumentParser(description="Query a PDF document interactively.")
    parser.add_argument("file_path", type=str, help="The path to the PDF file.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at '{args.file_path}'")
        return

    print("Setting up agent... This may take a moment.")
    try:
        # Instantiate the agent class, which holds its own state.
        doc_agent = DocumentAgent(file_path=args.file_path)
    except Exception as e:
        print(f"Failed to set up the agent. Error: {e}")
        return

    print("\n--- Document Analysis Agent ---")
    print(f"Setup complete. Talking about document: {os.path.basename(args.file_path)}")
    print("You can now ask questions. Type 'quit' or 'exit' to end the conversation.")

    chat_history = []

    while True:
        try:
            query = input("\nYour Question: ")

            if query.lower() in ["quit", "exit"]:
                print("Exiting. Goodbye!")
                break

            if not query:
                continue

            print("Agent is thinking...")

            # Call the agent's invoke method.
            result = doc_agent.invoke(query, chat_history)

            final_answer = result["output"]
            print("\nAssistant:")
            print(final_answer)

            # Update history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=final_answer))

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during conversation: {e}")

if __name__ == "__main__":
    main()