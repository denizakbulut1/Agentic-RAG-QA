# Document Analysis Agent

This project provides a conversational AI agent for analyzing and querying PDF documents. It features both a web-based interface and a command-line interface (CLI) for interacting with the agent. The agent is built using the LangChain library and OpenAI's language models.

## Features

*   **Document Upload and Analysis:** Upload PDF documents for analysis. The agent can classify documents (e.g., as a 'thesis' or 'paper') and analyze their structure.
*   **Conversational Q&A:** Ask questions about the document in a natural, conversational manner. The agent maintains a chat history to understand context.
*   **Section-Specific Analysis:** The agent can focus its analysis on specific sections of a document, which is particularly useful for large documents like theses.
*   **Web Interface:** An intuitive web interface for uploading documents and interacting with the agent.
*   **Command-Line Interface:** A CLI for users who prefer to work in a terminal environment.
*   **Caching:** The agent caches RAG chains and Table of Contents data to improve performance and reduce redundant processing.

## Project Structure

```
.
├── .gitignore
├── app.py
├── debug_imports.py
├── documents/
├── env/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── main_cli.py
│   ├── rag_core.py
│   └── utils.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
├── test_app.py
├── uploads/
└── vector_stores/
```

## Getting Started

### Prerequisites

*   Python 3.7+
*   An OpenAI API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

### Running the Application

#### Web Interface

To run the web application, use the following command:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

#### Command-Line Interface

To use the CLI, run the `main_cli.py` script with the path to your PDF file:

```bash
python -m src.main_cli "path/to/your/document.pdf"
```

## How It Works

The application uses a Retrieval-Augmented Generation (RAG) approach to answer questions about the documents. The core of the application is the `DocumentAgent`, which uses a ReAct-style agent to interact with the document.

### Agent and Tools

The `DocumentAgent` is equipped with a set of tools to perform various tasks:

*   **`classify_document_type`:** Determines if the document is a 'PhD thesis' or a 'scientific paper'.
*   **`analyze_thesis_structure`:** Analyzes a thesis to check if it's a collection of papers or a single monograph.
*   **`list_table_of_contents`:** Extracts and lists the table of contents from the document.
*   **`get_page_range_for_chapter`:**  Finds the start and end pages for a specific chapter.
*   **`answer_paper_question`:** Answers a question about the entire document (for papers).
*   **`answer_question_on_section`:** Answers a question about a specific section of a document (for theses).

### RAG Pipeline

1.  **Document Loading and Chunking:** The PDF document is loaded and split into smaller chunks of text using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
2.  **Vector Embeddings:** The text chunks are converted into vector embeddings using OpenAI's embedding models.
3.  **Vector Store:** The embeddings are stored in a FAISS vector store for efficient retrieval.
4.  **Conversational Chain:** A `ConversationalRetrievalChain` is created using the LangChain library. This chain uses the vector store to retrieve relevant document sections based on the user's query and the conversation history.
5.  **Response Generation:** The retrieved information is then used by a large language model (LLM) to generate a relevant and coherent answer.

## Key Components

*   **`app.py`:** The main Flask application file that defines the web routes and handles user requests.
*   **`src/main_cli.py`:** The entry point for the command-line interface.
*   **`src/agent.py`:** Contains the `DocumentAgent` class, which encapsulates the logic for interacting with the document. This class manages the agent's state, tools, and the agent executor.
*   **`src/rag_core.py`:**  Handles the core RAG-related tasks, such as creating the vector store and the conversational chain. It provides functions for creating RAG chains for both entire documents and specific sections.
*   **`src/utils.py`:** A utility module for managing the OpenAI API key.
*   **`templates/index.html`:** The main HTML file for the web interface.
*   **`static/`:** Contains the CSS and JavaScript files for the web interface.
*   **`test_app.py`:** A simple Flask application for testing purposes.
