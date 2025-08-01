# app.py

import os
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from src.agent import DocumentAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.agent import DocumentAgent

# Initialize the Flask application
app = Flask(__name__)

# Set a secret key for session management.
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
            session['filepath'] = filepath
            session['chat_history'] = [] # Initialize chat history for this session

            # Initial, automatic analysis
            initial_agent = DocumentAgent(file_path=filepath)
            initial_query = (
                "First, classify this document as a 'thesis' or a 'paper'. "
                "Then, if it is a thesis, analyze its structure to see if it's a compilation of papers. "
                "Provide a summary of your findings."
            )
            
            result = initial_agent.invoke(initial_query, [])
            initial_analysis = result.get('output', "Could not perform initial analysis.")
            
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
        # Internal caching will prevent re-processing the same file/section.
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