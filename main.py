# main.py
import streamlit as st
import os
import re
from dotenv import load_dotenv
from app.agent import run_triage_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="GitHub Issue Triage Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Helper Functions ---
def is_valid_github_url(url):
    """Check if the URL is a valid GitHub repository URL."""
    return re.match(r"https://github\.com/([^/]+)/([^/]+)", url)

def get_latest_issue_number(repo_url):
    """A mock function to get the latest issue number.
       In a real app, you'd use the GitHub API here.
       For this example, we'll just let the user input it.
    """
    
    return 1 

# --- UI Rendering ---
st.title("🤖 GitHub Issue Triage Agent")
st.caption("Automate the initial analysis of GitHub issues using an AI agent.")

# --- Prerequisite Checks ---
if not os.environ.get("OPENAI_API_KEY"):
    st.error("🚨 OpenAI API key not found. Please create a `.env` file and set your `OPENAI_API_KEY`.")
    st.stop()

if not os.path.exists("vector_store"):
    st.warning("⚠️ Vector store not found. The agent can't search local documentation.")
    if st.button("Build Documentation Vector Store"):
        with st.spinner("Processing documents in the `docs/` folder..."):
            os.system("python ingest.py")
        st.success("✅ Vector store created! The agent can now use the `search_documentation` tool.")
        st.rerun()
    st.info("To enable documentation search, add markdown files to the `docs/` folder and click the button above.")

st.markdown("---")

# --- Input Form ---
st.header("Triage a New Issue")
repo_url = st.text_input(
    "GitHub Repository URL",
    "https://github.com/langchain-ai/langgraph",
    help="Enter the full URL of the public GitHub repository."
)

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    issue_number = st.number_input(
        "Issue Number",
        min_value=1,
        step=1,
        help="Enter the specific issue number you want to triage."
    )

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    submit_button = st.button("✨ Run Triage", type="primary", use_container_width=True)


# --- Agent Execution ---
if submit_button:
    if not is_valid_github_url(repo_url):
        st.error("Invalid GitHub repository URL. Please use the format `https://github.com/owner/repo`.")
    else:
        with st.spinner(f"🕵️‍♂️ Agent is triaging issue #{issue_number}... This may take a minute."):
            try:
                report = run_triage_agent(repo_url, issue_number)
                st.subheader("📄 Triage Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")

st.markdown("---")
st.info("This agent uses LangGraph and multiple tools to analyze issues. It can fetch issue details, search the web, and search local documentation to provide a comprehensive triage report.")
