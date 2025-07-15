from typing import TypeDict, Annotated, Literal, List, Any, Dict
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypeDict):
    """Represents the state of agent"""
    repo_url: str
    issue_number: int
    issue_content: str
    triage_report: str
    messages: Annotated[List[BaseMessage], operator.add]


import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL = "sentence_transformers/allMiniLM-L6-v2"

def get_retriever():
    """create and returns retrievr from the local FAISS vector store"""
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Warning: Vector store not found. The 'search_documentation' tool will not work")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.from_documents(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k":3})
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None
    
import os
import re
from github import Github, GithubException
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearch
from agent.retriever import get_retriever

try:
    github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    g = Github(github_access_token) if github_access_token else Github()
except Exception as e:
    print(f"Could not initialize github client: {e}")
    g = None

web_search = DuckDuckGoSearch()

@tool
def get_github_issue(repo_url: str, issue_number: int) -> str:
    """Fetches the title and body of a specific Github issue from a Public repo"""
    print(f"----TOOL: Fetching Github Issue #{issue_number} from {repo_url}---")
    if not g:
        return "Github client NOt initialized"
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return "Invalid Github Repository URL format."
    
    owner, repo_name = match.groups()

    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
        issue = repo.get_issue(number=issue_number)
        return f"Issue Title: {issue.title}\n\nIssue Body:\n{issue.body}"
    except GithubException as e:
        return f"Error fetching issue: {e}. Please chek if the repository and issue number are correct"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    
@tool
def search_web_for_issue(query:str) -> str:
    """Searches the web using duckduckgo to find the solutions, discussions or context related to the github issue"""
    print(f"---TOOL: Searching web for: {query}---")
    return web_search(query)

@tool
def search_documentation(query:str) -> str:
    """Searches the project's own documentation (from the vector store) for relevant information"""
    print(f"---TOOL: Searching documentation for: {query}---")
    retriever = get_retriever()
    if retriever is None:
        return "Documentation retriever not available"
    
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the project documentation"
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

tools = [get_github_issue, search_documentation, search_web_for_issue]

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import AgentState

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    print("---AGENT NODE---")
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    print("---ROUTING: Checking for tool calls---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print(">>>Decision: Continue with tool execution")
        return "continue"
    print(">>>Decision: End of Process")
    return "end"

def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()

agent_executor = build_agent_graph()

def run_triage_agent(repo_url: str, issue_number: int):
    print(f"\n\n---Starting Triage for {repo_url} Issue #{issue_number}")

    prompt = f"""
    You are a senior software engineer responsible for triaging GitHub issues.
    Your goal is to analyze the issue, gather context, and produce a concise "Triage Report".

    Here is the task:
    1.  Start by using the `get_github_issue` tool to fetch the content of issue #{issue_number} from the repository at `{repo_url}`.
    2.  Based on the issue content, decide if you need more information.
        - If it seems like a bug, use the `search_web_for_issue` tool to look for similar error messages or discussions.
        - If it's a feature request or a question about usage, use the `search_documentation` tool to see if the project's own docs have an answer.
    3.  After gathering all necessary information, synthesize your findings.
    4.  Conclude your work by providing a final "Triage Report". This report should be a single, comprehensive message and should NOT call any more tools. The report must include:
        - **Summary:** A one-sentence summary of the issue.
        - **Suggested Labels:** A list of 1-3 suggested labels from this list: `bug`, `feature-request`, `question`, `documentation`, `needs-more-info`.
        - **Analysis & Context:** A brief paragraph explaining your reasoning, mentioning any relevant info you found from web or documentation searches.
        - **Next Steps:** A suggested next action for the development team (e.g., "Confirm the bug on a clean environment," "Clarify the feature requirements with the user," "Close as a duplicate of #...").

    Begin the process now.
    """
    initial_state = {"messages": [HumanMessage(content=prompt)]}
    
    final_state = agent_executor.invoke(initial_state, {"recursion_limit": 10})
    
    final_report = final_state["messages"][-1].content
    print(f"--- Triage Complete ---")
    return final_report
