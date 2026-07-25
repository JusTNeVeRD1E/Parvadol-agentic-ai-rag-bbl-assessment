# Agentic AI RAG System — Bangkok Bank AI Engineer Test

An Agentic AI system utilizing multi-agent orchestration combined with Retrieval-Augmented Generation (RAG). Developed by **Parvadol Kiratipongvut** as part of the AI Engineer Programming Test for Bangkok Bank.

---

## Project Overview

This project demonstrates a sequential multi-agent workflow designed to retrieve accurate context from a local knowledge base and synthesize it into well-formatted answers with low-latency execution and high language consistency.

### Agent Workflow
1. **Guardrail Check (Security Layer):**
   - **Role:** Screens user input using `meta-llama/llama-prompt-guard-2-86m` to prevent Prompt Injections, Jailbreaks, or malicious commands.
2. **Data Retriever Agent (Agent 1):**
   - **Role:** Extracts relevant search keywords and retrieves raw context snippets from `knowledge_base.txt` using a custom Python search tool.
   - **Output:** Raw contextual snippets and document sections.
3. **Report Generator Agent (Agent 2):**
   - **Role:** Synthesizes raw snippets into clean, professional, non-redundant, and structured responses (with support for Markdown Tables and automatic language matching).
   - **Output:** Final user-ready answer.

---

## Model Selection & Comparison (LLM Evaluation)

To deliver extremely fast responses (Ultra-low TTFT) while avoiding Free Tier Token Per Minute (TPM) bottlenecks, **LLaMA 3.1 8B Instant (Groq Cloud)** was selected as the core engine. Below is a detailed comparison against other leading models:

### Model Comparison Table

| Criteria | GPT-4o | Gemini 2.5 Flash | Claude 3 Opus | LLaMA 3.1 8B Instant (Groq) |
| :--- | :---: | :---: | :---: | :---: |
| **Provider / Company** | OpenAI | Google | Anthropic | Meta |
| **Access Method** | Native API | Native API | Native API | Third-Party API (Groq Cloud) |
| **Context Window (Max)** | 128K Tokens | 1M-2M Tokens | 200K Tokens | 128K Tokens |
| **Processing Speed (TTFT)** | 0.3 - 0.5 s | 0.3 - 0.5 s | 0.6 - 0.9 s | **< 0.2 s** |

> **Key Decision Takeaway:** Groq’s LLaMA 3.1 8B Instant achieves a Time To First Token (TTFT) under **0.2 seconds**, providing the ideal real-time streaming experience required for multi-agent RAG applications while maintaining high context retention (128K window) and generous TPM quotas (131,072 TPM).

---

## Sample Execution & Output Results

Below are demonstration samples of the system processing various employee policy queries, displaying automatic Markdown table rendering, multi-agent orchestration, and prompt security checks.

### 1. System Overview & UI Interface
The system features a clean Gradio interface integrated with real-time streaming output.

![System Architecture & Model Comparison](assets/model-comparison-table.png)

### 2. Policy Query with Automatic Markdown Table Rendering
When users ask for complex policy details (e.g., flight class rules or daily per diem rates), Agent 2 automatically structures the response into Markdown Tables without using emojis.

![Sample Execution - Table Response](assets/sample-execution-table-output.png)

### 3. Prompt Injection Security Block
If a prompt injection or jailbreak attempt is detected, the Guardrail immediately blocks execution prior to agent triggering.

![Sample Execution - Security Block](assets/sample-execution-security-block.png)

---
