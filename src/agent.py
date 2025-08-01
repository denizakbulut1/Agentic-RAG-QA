# src/agent.py

import json
import re
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_react_agent
from src.rag_core import create_qa_chain, create_qa_chain_for_section
from src.utils import OPENAI_API_KEY
from langchain_core.exceptions import OutputParserException

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
        self.structure_analysis_cache = None
        print(f"DocumentAgent initialized for: {self.file_path}")
        print("RAG chain cache is ready for this session.")
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", openai_api_key=OPENAI_API_KEY, temperature=0)
        self.agent_executor = self._setup_agent_executor()

    #Tool Methods

    def _analyze_thesis_structure(self, _: str) -> str:
        """
        Analyzes the thesis's Table of Contents to determine if it's a monograph or a 
        compilation of papers by looking for non-generic chapter titles.
        """
        if self.structure_analysis_cache:
            print("--- Retrieving thesis structure analysis from cache ---")
            return self.structure_analysis_cache
            
        print("--- TOOL: Analyzing thesis structure based on chapter titles ---")
        toc = self._get_table_of_contents()
        
        if isinstance(toc, str):
            return f"Cannot analyze structure, ToC could not be parsed: {toc}"

        # List of common, non-paper chapter titles
        generic_titles = [
            "introduction", "summary", "conclusion", "discussion", "background",
            "literature review", "methodology", "methods", "references", "bibliography",
            "acknowledgements", "abstract"
        ]

        identified_papers = []
        for item in toc:
            title = item.get("title", "").lower().strip()
            
            # Check if the title is generic
            is_generic = any(gen_title in title for gen_title in generic_titles)
            
            # A chapter is likely a paper if its title is not generic and is reasonably long
            if not is_generic and len(title) > 15:
                # Use the original title for display
                identified_papers.append(item.get("title", "Unknown Title"))

        if not identified_papers:
            result = "Analysis complete: This document appears to be a standard monograph-style thesis, as most chapters have generic titles like 'Introduction' or 'Conclusion'."
        else:
            papers_list = "\n- ".join(identified_papers)
            result = (
                f"Analysis complete: This thesis appears to be a compilation of {len(identified_papers)} papers. "
                "The following non-generic chapters have been identified as potential standalone papers:\n"
                f"- {papers_list}"
            )
        
        self.structure_analysis_cache = result
        return result
    
    def _classify_document_type(self, _: str) -> str:
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
    
    
    def _get_table_of_contents(self) -> list | str:
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
        if not toc_text.strip(): return "No text could be extracted to find a ToC."
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
    
    def _list_table_of_contents(self, _: str) -> str:
        print("--- TOOL: Listing Table of Contents ---")
        toc = self._get_table_of_contents()
        if isinstance(toc, str): return f"Could not list ToC: {toc}"
        if not toc: return "The Table of Contents is empty or could not be found."
        formatted_list = [f"{i+1}. {item.get('title', 'N/A')} (page {item.get('page', 'N/A')})" for i, item in enumerate(toc)]
        return "\n".join(formatted_list)

    def _get_page_range_for_chapter(self, chapter_identifier: str) -> str:
        print(f"--- TOOL: Getting page range for '{chapter_identifier}' ---")
        toc = self._get_table_of_contents()
        if isinstance(toc, str): return f"Could not get page range because ToC could not be parsed: {toc}"
        if not toc: return "Could not find a table of contents to search."
        norm_identifier = chapter_identifier.lower().replace("chapter", "").strip()
        found_chapter_index = -1
        for i, item in enumerate(toc):
            norm_title = item.get("title", "").lower()
            if norm_identifier in norm_title or re.match(f"^{re.escape(norm_identifier)}[ .:]", norm_title):
                found_chapter_index = i
                break
        if found_chapter_index == -1: return f"Could not find a chapter matching '{chapter_identifier}' in the Table of Contents."
        start_page = toc[found_chapter_index]["page"]
        if found_chapter_index + 1 < len(toc): end_page = toc[found_chapter_index + 1]["page"] - 1
        else: end_page = start_page + 100
        result = {"start_page": start_page, "end_page": end_page}
        return json.dumps(result)
    
    def _answer_paper_question(self, query: str) -> str:
        print(f"--- TOOL: Answering question for entire paper. Query: '{query}' ---")
        cache_key = self.file_path
        if cache_key not in self.rag_chain_cache:
            print("--- RAG chain not in cache. Creating and caching... ---")
            self.rag_chain_cache[cache_key] = create_qa_chain(self.file_path)
        qa_chain = self.rag_chain_cache[cache_key]
        result = qa_chain.invoke({"question": query})
        return result.get("answer", "No answer could be generated.")
    
    def _answer_question_on_section(self, tool_input: str) -> str:
        try:
            params = json.loads(tool_input)
            query, start_page, end_page = params["query"], int(params["start_page"]), int(params["end_page"])
            print(f"--- TOOL: Answering question on section (p{start_page}-{end_page}). Query: '{query}' ---")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return f"Error: Invalid input format. Expected JSON with 'query', 'start_page', 'end_page'. Details: {e}"
        cache_key = f"{self.file_path}_{start_page}_{end_page}"
        if cache_key not in self.rag_chain_cache:
            print(f"--- RAG chain for section p{start_page}-{end_page} not in cache. Creating... ---")
            try:
                self.rag_chain_cache[cache_key] = create_qa_chain_for_section(self.file_path, start_page, end_page)
            except Exception as e: return f"Failed to create RAG chain for section: {e}"
        qa_chain = self.rag_chain_cache[cache_key]
        result = qa_chain.invoke({"input": query})
        return result.get("answer", "No answer could be generated for the specified section.")
    
    def _setup_agent_executor(self):
        tools = [
            Tool(name="classify_document_type", func=self._classify_document_type, description="Determines if the document is a 'PhD thesis' or a 'scientific paper'."),
            Tool(name="analyze_thesis_structure", func=self._analyze_thesis_structure, description="Analyzes a thesis to check if it's a collection of papers or a single monograph based on chapter titles."),
            Tool(name="list_table_of_contents", func=self._list_table_of_contents, description="Gets a numbered list of all chapter titles and their start pages."),
            Tool(name="get_page_range_for_chapter", func=self._get_page_range_for_chapter, description="Gets the exact start/end pages for a chapter with a known title or number."),
            Tool(name="answer_paper_question", func=self._answer_paper_question, description="Answers a specific question about a document classified as a 'paper'."),
            Tool(name="answer_question_on_section", func=self._answer_question_on_section, description="Answers a question about a specific section of a thesis using a page range."),
        ]
        
        # This prompt is proven to work well for preventing loops.
        system_prompt = """You are a ReAct-style agent. You must follow the response format precisely.

You have access to the following tools:
{tools}

**RESPONSE FORMAT (CRITICAL):**
After every `Thought:`, you must choose one of two options:

**OPTION 1: Use a tool to gather more information.**
The format MUST be a three-line block:

Thought: [Your reasoning to use a tool]
Action: [tool_name]
Action Input: [The input for the tool. Use an empty string "" if the tool takes no input.]


**OPTION 2: Give the final answer because you have enough information.**
The format MUST be a two-line block:

Thought: [Your reasoning that you have the final answer]
Final Answer: [The direct answer to the user's question]


**EXAMPLE: The user asks "is this a paper?"**
*Your first response:*

Thought: I need to know the document type to answer the question. I will use the classify tool.
Action: classify_document_type
Action Input: ""

*(After this, you will get an Observation, e.g., "thesis")*

*Your second response:*

Thought: The previous tool call returned 'thesis'. I now know the document is not a paper. I have enough information to give the final answer.
Final Answer: No, this document is a thesis, not a paper.


**Operational Rules:**
1.  Always start by classifying the document if the type is unknown.
2.  For a thesis, if asked about its composition (e.g., "are there sub-papers?"), use `analyze_thesis_structure`.
3.  Use the other tools as needed, but always aim to reach a `Final Answer`.

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
        """The main entry point for the user to interact with the agent."""
        return self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history,
            "file_path": self.file_path
        })

