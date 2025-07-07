# app.py

import os
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from src.agent import DocumentAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# We can now import your agent directly
from src.agent import DocumentAgent

# Initialize the Flask application
app = Flask(__name__)

# Set a secret key for session management.
# This is crucial for keeping user-specific data separate.
app.secret_key = os.urandom(24) 

# Configure a temporary upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Routes ---

@app.route('/')
def index():
    """Render the main user interface page."""
    # Clear any old session data when the user loads the main page
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document uploads and initialize the agent for the session."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # --- AGENT INITIALIZATION ---
            # Instantiate the agent for this specific file.
            # We will store the agent object in the session.
            # NOTE: Storing complex objects in session is not ideal for production,
            # but it is PERFECT for a single-user demo.
            # The agent itself will cache the RAG chains.
            
            # For the demo, we'll store the filepath and re-instantiate the agent on each call.
            # This is simpler and avoids session serialization issues with complex objects.
            session['filepath'] = filepath
            session['chat_history'] = [] # Initialize chat history for this session

            # We can perform an initial, automatic analysis here.
            # Let's ask the agent to classify the doc and analyze its structure.
            initial_agent = DocumentAgent(file_path=filepath)
            initial_query = (
                "First, classify this document as a 'thesis' or a 'paper'. "
                "Then, if it is a thesis, analyze its structure to see if it's a compilation of papers. "
                "Provide a summary of your findings."
            )
            
            # We don't need chat history for the very first query.
            result = initial_agent.invoke(initial_query, [])
            initial_analysis = result.get('output', "Could not perform initial analysis.")
            
            # The agent executor's verbose output is great for the terminal, but too noisy for the UI.
            # We only send the final answer ('output').
            return jsonify({
                "message": f"Successfully processed '{filename}'",
                "initial_analysis": initial_analysis
            })
            
        except Exception as e:
            # Clean up the session if initialization fails
            session.clear()
            # Providing a detailed error message is helpful for debugging
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    return jsonify({"error": "File upload failed"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions for the currently active document."""
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Check if a document has been uploaded for this session
    if 'filepath' not in session:
        return jsonify({"error": "Please upload a document first."}), 400

    try:
        # --- AGENT INVOCATION ---
        # Re-instantiate the agent with the filepath from the session.
        # Your agent's internal caching will prevent re-processing the same file/section.
        doc_agent = DocumentAgent(file_path=session['filepath'])
        
        # Get the current chat history from the session
        chat_history = session.get('chat_history', [])

        # Call the agent's invoke method
        result = doc_agent.invoke(question, chat_history)
        final_answer = result.get("output", "No answer could be generated.")

        # Update chat history in the session
        # Note: Your HumanMessage/AIMessage objects might not be JSON serializable.
        # Storing as simple dicts is safer for sessions.
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": final_answer})
        session['chat_history'] = chat_history
        
        return jsonify({"answer": final_answer})
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # To run: `python app.py`
    # Access at http://127.0.0.1:5000
    app.run(debug=False, port=5000)