# Agentic AI RAG System — Bangkok Bank AI Engineer Test

An Agentic AI system utilizing multi-agent orchestration combined with Retrieval-Augmented Generation (RAG). Developed by **Parvadol Kiratipongvut** as part of the AI Engineer Programming Test for Bangkok Bank.

---

## Project Overview

This project demonstrates a sequential multi-agent workflow designed to retrieve accurate context from a local knowledge base and synthesize it into well-formatted answers.

### Agent Workflow
1. **Data Retriever Agent (RAG):**
   - **Role:** Extracts relevant raw text chunks from `knowledge_base.txt` using a custom Python search utility.
   - **Output:** Raw contextual text snippets.
2. **Report Generator Agent:**
   - **Role:** Processes raw snippets from the retriever and synthesizes them into a cohesive, clear, non-redundant response.
   - **Output:** Final formatted answer for the end-user.

---

## Repository Structure

```text
.
├── knowledge_base.txt    # Local knowledge base containing sample policy data
├── main.py               # Core application logic & agent orchestration
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── screenshots/          # Execution outputs for sample test queries
