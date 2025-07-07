# src/agent.py (With new list_table_of_contents tool)

import json
import re
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_react_agent
from src.rag_core import create_qa_chain, create_qa_chain_for_section
from src.utils import OPENAI_API_KEY
from langchain_core.exceptions import OutputParserException

# This error handler is perfect, no changes needed.
def _handle_parsing_error(error: OutputParserException) -> str:
    response = (
        "I'm sorry, my previous response was not formatted correctly. "
        "I must always use the ReAct format with a 'Thought:', and then an 'Action:' or 'Final Answer:'.\n\n"
        f"Here was my faulty response: ```{error.llm_output}```\n"
        f"Here is the error from the parser: {error.observation}\n\n"
        "I will try again, making sure to use the correct format."
    )
    return response

class DocumentAgent:
    def __init__(self, file_path: str):
        if not file_path:
            raise ValueError("A file path must be provided.")
        self.file_path = file_path
        self.rag_chain_cache = {}
        self.toc_cache = None
        print(f"DocumentAgent initialized for: {self.file_path}")
        print("RAG chain cache is ready for this session.")
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key=OPENAI_API_KEY, temperature=0)
        self.agent_executor = self._setup_agent_executor()

    # --- Tool Methods ---

    # No changes to _classify_document_type
    def _classify_document_type(self, _: str) -> str:
        # ... (same as before)
        print("--- TOOL: Classifying document type ---")
        extracted_text = ""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= 3: break
                    extracted_text += page.extract_text() or ""
        except Exception as e:
            return f"Error reading PDF for classification: {e}"
        if not extracted_text.strip(): return "unknown"
        classifier_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Is the text from a 'thesis' or a 'paper'? Respond with ONLY 'thesis' or 'paper'."),
            ("human", extracted_text)
        ])
        chain = prompt | classifier_model
        doc_type = chain.invoke({}).content.strip().lower()
        return doc_type if doc_type in ["thesis", "paper"] else "unknown"

    # _get_table_of_contents helper remains the same internal function
    def _get_table_of_contents(self) -> list | str:
        # ... (same as before)
        if self.toc_cache is not None:
            print("--- Retrieving ToC from cache ---")
            return self.toc_cache
        print("--- Parsing document for Table of Contents ---")
        toc_text = ""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                pages_to_scan = min(20, len(pdf.pages))
                for i in range(pages_to_scan):
                    toc_text += pdf.pages[i].extract_text(x_tolerance=1, y_tolerance=3) or ""
        except Exception as e:
            return f"Error reading PDF for ToC: {e}"
        if not toc_text.strip():
            return "No text could be extracted to find a ToC."
        parser_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "You are a text-processing utility. Analyze the following text and extract the Table of Contents. "
            "Respond with ONLY a valid JSON array, where each object has a 'title' (string) and 'page' (integer) key. "
            "Example: [{{\"title\": \"Chapter 1: Introduction\", \"page\": 1}}, {{\"title\": \"Chapter 2: Background\", \"page\": 15}}]\n\n"
            "Text: {text}"
        )
        chain = prompt | parser_model
        response_content = chain.invoke({"text": toc_text}).content
        try:
            json_str = response_content.strip().replace("`", "")
            if json_str.startswith("json"): json_str = json_str[4:]
            parsed_toc = json.loads(json_str)
            self.toc_cache = parsed_toc
            return parsed_toc
        except json.JSONDecodeError:
            error_msg = f"Failed to parse ToC into JSON. Raw response: {response_content}"
            self.toc_cache = error_msg
            return error_msg

    # --- NEW TOOL METHOD: _list_table_of_contents ---
    def _list_table_of_contents(self, _: str) -> str:
        """
        Gets the table of contents and formats it as a human-readable string for the agent.
        """
        print("--- TOOL: Listing Table of Contents ---")
        toc = self._get_table_of_contents()
        
        if isinstance(toc, str): # Handle error case
            return f"Could not list ToC: {toc}"
        
        if not toc:
            return "The Table of Contents is empty or could not be found."
            
        # Format the JSON into a nice string for the LLM to read
        formatted_list = []
        for i, item in enumerate(toc):
            title = item.get('title', 'N/A')
            page = item.get('page', 'N/A')
            formatted_list.append(f"{i+1}. {title} (page {page})")
            
        return "\n".join(formatted_list)

    # _get_page_range_for_chapter and the other tools remain the same
    def _get_page_range_for_chapter(self, chapter_identifier: str) -> str:
        # ... (same as before)
        print(f"--- TOOL: Getting page range for '{chapter_identifier}' ---")
        toc = self._get_table_of_contents()
        if isinstance(toc, str):
            return f"Could not get page range because ToC could not be parsed: {toc}"
        if not toc:
            return "Could not find a table of contents to search."
        norm_identifier = chapter_identifier.lower().replace("chapter", "").strip()
        found_chapter_index = -1
        for i, item in enumerate(toc):
            norm_title = item.get("title", "").lower()
            if norm_identifier in norm_title or re.match(f"^{re.escape(norm_identifier)}[ .:]", norm_title):
                found_chapter_index = i
                break
        if found_chapter_index == -1:
            return f"Could not find a chapter matching '{chapter_identifier}' in the Table of Contents."
        start_page = toc[found_chapter_index]["page"]
        if found_chapter_index + 1 < len(toc):
            end_page = toc[found_chapter_index + 1]["page"] - 1
        else:
            end_page = start_page + 100
        result = {"start_page": start_page, "end_page": end_page}
        return json.dumps(result)

    def _answer_paper_question(self, query: str) -> str:
        # ... (same as before)
        print(f"--- TOOL: Answering question for entire paper. Query: '{query}' ---")
        cache_key = self.file_path
        if cache_key not in self.rag_chain_cache:
            print("--- RAG chain not in cache. Creating and caching... ---")
            self.rag_chain_cache[cache_key] = create_qa_chain(self.file_path)
        qa_chain = self.rag_chain_cache[cache_key]
        result = qa_chain.invoke({"input": query})
        return result.get("answer", "No answer could be generated.")

    def _answer_question_on_section(self, tool_input: str) -> str:
        # ... (same as before)
        try:
            params = json.loads(tool_input)
            query = params["query"]
            start_page = int(params["start_page"])
            end_page = int(params["end_page"])
            print(f"--- TOOL: Answering question on section (p{start_page}-{end_page}). Query: '{query}' ---")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return f"Error: Invalid input format for tool. Expected a JSON string with 'query', 'start_page', and 'end_page'. Details: {e}"
        cache_key = f"{self.file_path}_{start_page}_{end_page}"
        if cache_key not in self.rag_chain_cache:
            print(f"--- RAG chain for section p{start_page}-{end_page} not in cache. Creating... ---")
            try:
                self.rag_chain_cache[cache_key] = create_qa_chain_for_section(self.file_path, start_page, end_page)
            except Exception as e:
                return f"Failed to create RAG chain for section: {e}"
        qa_chain = self.rag_chain_cache[cache_key]
        result = qa_chain.invoke({"input": query})
        return result.get("answer", "No answer could be generated for the specified section.")


    def _setup_agent_executor(self):
        # --- UPDATED TOOL LIST ---
        tools = [
            Tool(
                name="classify_document_type",
                func=self._classify_document_type,
                description="Use this FIRST to determine if the document is a 'PhD thesis' or a 'scientific paper'."
            ),
            # THE NEW TOOL IS ADDED HERE
            Tool(
                name="list_table_of_contents",
                func=self._list_table_of_contents,
                description="Use this to get a numbered list of all chapter titles and their start pages. This is useful for finding the name of a specific chapter."
            ),
            Tool(
                name="get_page_range_for_chapter",
                func=self._get_page_range_for_chapter,
                description="Use this to get the exact start and end page numbers for a chapter you already know the name of. Input should be the chapter title or number (e.g., 'Introduction' or '3')."
            ),
            Tool(
                name="answer_paper_question",
                func=self._answer_paper_question,
                description="Use this for a 'paper' to answer a specific question. The input is the user's full question."
            ),
            Tool(
                name="answer_question_on_section",
                func=self._answer_question_on_section,
                description="Use this for a 'thesis' ONLY AFTER you have the exact page range from 'get_page_range_for_chapter'. The input MUST be a JSON string with 'query', 'start_page', and 'end_page'."
            )
        ]
        
        # --- UPDATED SYSTEM PROMPT ---
        system_prompt = """You are a ReAct-style agent. Your primary goal is to answer questions about scientific documents by breaking the problem down into steps and using your tools.

You have access to the following tools:
{tools}

**Your Response Format:**
You MUST respond with a block of text starting with `Thought:`, followed by `Action:` and `Action Input:`, or `Final Answer:`.

**EXAMPLE 1: Finding a chapter name**
Thought: The user wants to know the name of the third chapter. To do this, I need to see the whole table of contents first.
Action: list_table_of_contents
Action Input: ""

**EXAMPLE 2: Summarizing a chapter**
Thought: The user wants a summary of the 'Results' chapter. First, I need to get the page range for that chapter.
Action: get_page_range_for_chapter
Action Input: "Results"

**Your Operational Rules:**
1.  **Check History First:** If the answer is already in the chat history, use it.
2.  **Classify First:** If the document type is unknown, your first action MUST be `classify_document_type`.
3.  **Thesis Workflow (CRITICAL):**
    a. To find the NAME of a chapter, use `list_table_of_contents`.
    b. To get page numbers for a known chapter to read its content, use `get_page_range_for_chapter`.
    c. To read the content of a chapter, use `answer_question_on_section` with the page range from the previous step.
    d. DO NOT try to guess page numbers or chapter names.

When providing an action, it MUST be one of the following: [{tool_names}]

Begin!"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "DOCUMENT PATH: {file_path}\n\nQUESTION: {input}"),
            ("ai", "{agent_scratchpad}"),
        ])
        
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=_handle_parsing_error
        )

    def invoke(self, query: str, chat_history: list):
        # ... (same as before)
        return self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history,
            "file_path": self.file_path
        })