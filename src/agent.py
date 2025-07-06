import json
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from src.rag_core import create_qa_chain, create_qa_chain_for_section
from langchain_core.prompts import MessagesPlaceholder
from src.utils import OPENAI_API_KEY

# --- Helper function to clean the agent's messy input ---
def extract_path_from_action_input(input_string: str) -> str:
    """
    A helper function to reliably extract the file path from the LLM's output.
    It handles cases like "file_path = 'C:\...'" or just "'C:\...'".
    """
    # The LLM often wraps the path in single quotes. We extract the content between them.
    if "'" in input_string:
        try:
            # Splits the string by the quote and takes the content in the middle
            return input_string.split("'")[1]
        except IndexError:
            # If splitting fails for some reason, return the original string after stripping whitespace
            return input_string.strip()
    # If there are no quotes, just return the cleaned string
    return input_string.strip()


# --- Your core functions for document processing ---
def classify_document_type(file_path: str) -> str:
    """Classifies a PDF as 'thesis' or 'paper'."""
    extracted_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= 3: break
                extracted_text += page.extract_text() or ""
    except Exception as e:
        print(f"Error in classify_document_type reading PDF: {e}")
        return "Error: Could not read PDF."
    if not extracted_text.strip(): return "unknown"

    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Is the following text from a 'thesis' or a 'paper'? Respond with ONLY the word 'thesis' or 'paper'."),
        ("human", extracted_text)
    ])
    chain = prompt | model
    try:
        doc_type = chain.invoke({}).content.strip().lower()
        if doc_type in ["thesis", "paper"]:
            return doc_type
        return "unknown"
    except Exception as e:
        print(f"Error in classify_document_type LLM call: {e}")
        return "Error: LLM classification failed."

def summarize_thesis(file_path: str) -> str:
    """Generates a high-level overview of a thesis."""
    extracted_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= 10: break
                extracted_text += page.extract_text() or ""
    except Exception as e:
        return f"Error reading PDF for summary: {e}"
    if not extracted_text.strip(): return "No content to summarize."

    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Provide a concise, 3-5 sentence high-level overview of the following PhD thesis text. Focus on the main research question, methodology, and key findings."),
        ("human", extracted_text)
    ])
    chain = prompt | model
    try:
        return chain.invoke({}).content
    except Exception as e:
        return f"Error generating summary: {e}"
    

def get_table_of_contents(file_path: str, pages_to_scan: int = 20) -> str:
    """
    Scans the first few pages of a PDF to find and extract the Table of Contents.

    Args:
        file_path: The path to the PDF file.
        pages_to_scan: The number of pages to scan from the beginning of the PDF.

    Returns:
        A formatted string of the Table of Contents, or a message if not found.
    """
    print(f"--- Scanning for Table of Contents in '{file_path}' ---")
    toc_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            # Limit the scan to the first few pages to save time and cost :)
            num_pages = pages_to_scan
            for i in range(num_pages):
                page = pdf.pages[i]
                toc_text += page.extract_text() or ""
                toc_text += "\n---\n" # Page separator
    except Exception as e:
        return f"Error: Could not read the PDF file to find the Table of Contents. Details: {e}"

    if not toc_text.strip():
        return "Error: No text could be extracted from the first few pages."

    # Use an LLM to parse the extracted text and find the ToC
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a text-processing utility that specializes in finding and formatting a Table of Contents (ToC).
    - Your task is to analyze the provided text, which is from the first few pages of a document.
    - **The heading might be 'Table of Contents', 'Contents', 'Inhoud', or something similar.**
    - Identify the list of chapters or sections and their page numbers.
    - Re-format it as a clean, easy-to-read list. Each item should have the chapter/section title and its page number.
    - If you absolutely cannot find a list of chapters and page numbers, you MUST respond with the exact phrase: 'No Table of Contents found.'
    - Do not add any commentary or introductory text like 'Here is the Table of Contents'. Respond only with the formatted ToC or the 'not found' message."""),
    ("human", f"Here is the text to analyze:\n\n```\n{toc_text}\n```")
    ])

    chain = prompt | model
    
    try:
        response = chain.invoke({})
        return response.content
    except Exception as e:
        return f"Error: The LLM failed while trying to parse the Table of Contents. Details: {e}"

def answer_paper_question(file_path: str, query: str) -> str:
    """Answers a specific question about a paper using RAG."""
    try:
        qa_chain = create_qa_chain(file_path)
        result = qa_chain.invoke({"input": query})
        return result["answer"]
    except Exception as e:
        return f"Error answering paper question: {e}"





def setup_agent_executor(file_path: str, rag_chain_cache: dict):
    """
    Sets up the conversational agent, now with a caching mechanism for RAG chains.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

    tools = [
        Tool(
            name="classify_document",
            func=lambda tool_input: classify_document_type(extract_path_from_action_input(tool_input)),
            description="Use this to determine if a document is a 'PhD thesis' or a 'scientific paper'. The input MUST be the full file path."
        ),
        Tool(
            name="get_table_of_contents",
            func=lambda tool_input: get_table_of_contents(extract_path_from_action_input(tool_input)),
            description="Use this on a thesis to get its structure (chapters and page numbers). The input MUST be the full file path."
        ),
        Tool(
            name="summarize_thesis",
            func=lambda tool_input: summarize_thesis(extract_path_from_action_input(tool_input)),
            description="Use this to get a high-level overview of a thesis. The input MUST be the full file path."
        ),
        # --- MODIFIED TOOL 1 ---
        Tool(
            name="answer_paper_question",
            func=lambda query: (
                # Check cache first
                rag_chain_cache[file_path]
                if file_path in rag_chain_cache 
                # If not in cache, create it, store it, and then return it
                else rag_chain_cache.setdefault(file_path, create_qa_chain(file_path=file_path))
            ).invoke({"input": query})["answer"],
            description="Use this for a document classified as a 'paper' to answer a specific question. The input is the user's question."
        ),
        # --- MODIFIED TOOL 2 ---
        Tool(
            name="answer_question_on_section",
            func=lambda tool_input: (
                lambda fp, q, sp, ep: (
                    # Create a unique key for the section
                    (cache_key := f"{fp}_{sp}_{ep}") and
                    # Check cache first
                    (rag_chain_cache[cache_key]
                    if cache_key in rag_chain_cache
                    # If not in cache, create it, store it, and then return it
                    else rag_chain_cache.setdefault(cache_key, create_qa_chain_for_section(file_path=fp, start_page=sp, end_page=ep)))
                ).invoke({"input": q})["answer"]
            )(file_path, json.loads(tool_input)["query"], int(json.loads(tool_input)["start_page"]), int(json.loads(tool_input)["end_page"])),
            description="Use this for a 'thesis' to answer a question about a specific section ONLY after getting the page numbers from the table of contents. The input MUST be a JSON string with 'query', 'start_page', and 'end_page' keys."
        )
    ]


    # --- THE DEFINITIVE PROMPT FIX ---

    # 1. Define the system message string with the REQUIRED placeholders: {tools} and {tool_names}.
    # We are NOT using an f-string here. These are literal placeholders for the agent.
    system_message_string = """You are a ReAct-style agent designed to analyze scientific documents. You must follow a strict, sequential reasoning process.

You have access to the following tools:
{tools}

**Your Operational Rules:**
1.  **One Step at a Time:** You must break down problems into single, atomic steps. Do not try to reason about the content of a chapter and find its page numbers in the same thought.
2.  **Check History First (CRITICAL RULE):** Before using any tool, you MUST review the `chat_history`. If the user's current `input` asks for information that has already been provided, you MUST answer directly from the history. Do not use a tool to get the same information twice.
3.  **Classify if Needed:** If the document type is unknown (not in history), your next action MUST be `classify_document`.
4.  **Follow the Correct Path:** Once the document is classified, you must follow the correct workflow for 'paper' or 'thesis' to answer questions not found in the history.
5.  **Strict Output Format:** Your response MUST start with `Thought:` and follow the ReAct format precisely. After a thought, you must provide EITHER an `Action:` OR a `Final Answer:`, but NEVER both.

You have been provided with examples below. Follow these patterns precisely.

--- EXAMPLE 0: SIMPLE CLASSIFICATION QUESTION ---
Thought: The user is asking if the document is a thesis. The chat history is empty. My first step should be to classify the document using the `classify_document` tool.
Action: classify_document
Action Input: C:\\Path\\to\\file.pdf
Observation: thesis
Thought: The tool's observation is 'thesis'. I can now directly answer the user's question.
Final Answer: Yes, the document is a thesis.
--- END EXAMPLE 0 ---

--- EXAMPLE 1: HANDLING A SCIENTIFIC PAPER ---
Thought: The user is asking about the research methods used. First, I need to know the document type. The chat history is empty, so I must use the `classify_document` tool.
Action: classify_document
Action Input: C:\\Path\\to\\paper.pdf
Observation: paper
Thought: The document is a 'paper'. According to my rules, I should use the `answer_paper_question` tool to answer the user's specific question.
Action: answer_paper_question
Action Input: What were the research methods used in this paper?
Observation: The study employed a quantitative cross-sectional design, utilizing surveys distributed to 500 participants.
Thought: I have found the answer using the tool. I can now provide the final answer.
Final Answer: The research method used was a quantitative cross-sectional design with surveys.
--- END EXAMPLE 1 ---

--- EXAMPLE 2: HANDLING A PHD THESIS (FOLLOW-UP QUESTION) ---
Thought: The user is asking for the name of the fourth chapter. I will check the chat history first. The history shows the document has already been classified as a 'thesis'. My rules for a specific thesis question say I must use `get_table_of_contents` to find the chapter structure.
Action: get_table_of_contents
Action Input: C:\\Path\\to\\thesis.pdf
Observation: Contents
1. Introduction ...................................... 1
2. Background .................................... 15
3. Methodology ................................... 30
4. A Novel Approach to X ......................... 55
5. Conclusion .................................... 90
Thought: The tool returned the Table of Contents. I can clearly see that the fourth chapter is 'A Novel Approach to X'. I have all the information needed to answer the user's question.
Final Answer: The name of the fourth chapter is 'A Novel Approach to X'.
--- END EXAMPLE 2 ---

--- EXAMPLE 3 (REVISED FOR PATIENCE): HANDLING A COMPLEX THESIS QUESTION ---
Thought: The user is asking for a summary of the 'methodology' chapter. The document is a 'thesis'. My first step is to find the page numbers for that chapter. I will use the `get_table_of_contents` tool for this.
Action: get_table_of_contents
Action Input: C:\\Path\\to\\thesis.pdf
Observation: ... 3. Methodology ................................... 30 ... 4. A Novel Approach to X ......................... 55 ...
Thought: I have the Table of Contents. From the observation, I can see the 'Methodology' chapter starts on page 30 and the next chapter starts on page 55. This is the information I needed. Now, my next step is to use the `answer_question_on_section` tool to read the content of these pages and answer the user's question.
Action: answer_question_on_section
Action Input: {{"query": "Summarize the findings from this section", "start_page": 30, "end_page": 55}}
Observation: The section details a mixed-methods approach, combining qualitative interviews with a quantitative survey...
Thought: I have successfully used the tool to get a summary of the requested section. I can now give the final answer.
Final Answer: The methodology chapter (pages 30-55) describes a mixed-methods approach that combines qualitative interviews with a quantitative survey.
--- END EXAMPLE 3 ---

--- EXAMPLE 4 (IMPROVED): RECALLING FROM HISTORY WITH DIFFERENT PHRASING ---
# This example teaches you to find answers in the history even if the new question isn't identical.

# CONTEXT: The chat history already contains the following exchange:
# Human: "What is the name of the fourth chapter?"
# AI: "The name of the fourth chapter is 'A Novel Approach to X'."

# Now, the user asks a similar question:
Thought: The user is asking for the title of chapter 4. I must check the `chat_history` first. I see that I have already answered a question about the name of the fourth chapter. The previous answer was 'A Novel Approach to X'. This is the same information the user wants now. I do not need to use a tool.
Final Answer: As we discussed, the fourth chapter is titled 'A Novel Approach to X'.
--- END EXAMPLE 4 ---

When providing an action, it MUST be one of the following: [{tool_names}]

Begin! You will be given the user's `input` and the `chat_history`. Your response must start with `Thought:`.
"""

    # 2. Create the final ChatPromptTemplate.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message_string),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "DOCUMENT PATH: {file_path}\n\nQUESTION: {input}"),
        # This remains the key fix for the runtime error.
        ("ai", "{agent_scratchpad}"),
    ])

    # 3. Create the agent. This will now succeed because the 'prompt' object
    #    has the '{tools}' and '{tool_names}' placeholders it's looking for.
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    return agent_executor

