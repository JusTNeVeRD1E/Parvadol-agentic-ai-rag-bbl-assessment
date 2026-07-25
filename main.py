import os
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
print("--- Initializing Secure Multi-Agent RAG System (Groq Powered) ---")

groq_client = Groq(api_key=groq_api_key)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.1,
)

# ==============================================================================
#  GUARDRAIL: PROMPT INJECTION & JAILBREAK CHECKER
# ==============================================================================
def check_prompt_safety(user_input: str) -> bool:
    """
    Uses Llama-Prompt-Guard-2-86m to detect Prompt Injections, Jailbreaks, or Malicious Input.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-prompt-guard-2-86m",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.0,
            max_completion_tokens=10,
            top_p=1,
            stream=False,
        )
        output_text = completion.choices[0].message.content.strip().lower()
        if "unsafe" in output_text or "jailbreak" in output_text:
            return False
        return True
    except Exception as e:
        print(f" Prompt Guard Warning: {e}")
        return True


# ==============================================================================
# 1. ENHANCED CUSTOM TOOL FOR AGENT 1 (THAI & ENG SUPPORT)
# ==============================================================================
@tool
def search_knowledge_base(search_keyword: str) -> str:
    """
    Custom Python Tool that reads knowledge_base.txt ON-DEMAND and
    performs an improved multi-keyword relevance search to retrieve snippets.
    Supports both Thai and English search keywords.
    """
    file_path = "knowledge_base.txt"
    if not os.path.exists(file_path):
        return f"Error: Knowledge base file '{file_path}' not found."

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = [s.strip() for s in content.split("\n\n") if s.strip()]

    clean_keyword = (
        search_keyword.lower()
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
    )
    
    stop_words = {
        "what", "is", "the", "policy", "on", "for", "of", "and", "in", 
        "to", "a", "an", "are", "does", "about", "allowance", "class",
    }
    
    keywords = [
        w for w in clean_keyword.split() if len(w) > 1 and w not in stop_words
    ]

    if not keywords:
        keywords = [w for w in clean_keyword.split() if len(w) > 1]

    scored_sections = []
    for sec in sections:
        sec_lower = sec.lower()
        score = sum(1 for kw in keywords if kw in sec_lower)
        if score > 0:
            scored_sections.append((score, sec))

    scored_sections.sort(key=lambda x: x[0], reverse=True)

    if scored_sections:
        top_snippets = [sec for _, sec in scored_sections[:3]]
        return "\n\n---\n\n".join(top_snippets)

    return (
        "\n\n---\n\n".join(sections[1:5]) if len(sections) > 1 else content[:2000]
    )


# ==============================================================================
# 2. AGENT 1: Data Retriever Agent (Bilingual Support)
# ==============================================================================
llm_with_tools = llm.bind_tools([search_knowledge_base])

agent1_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Agent 1 (Data Retriever Agent).
Your ONLY role is to extract precise key terms and retrieve relevant information snippets from the knowledge base using the `search_knowledge_base` tool.

INSTRUCTIONS:
1. Extract core English search terms from the user's query (e.g., "what is the policy on flight allowance?" -> Key terms: "flight allowance policy").
2. Call `search_knowledge_base` with these concise key terms.
3. Return ONLY raw retrieved snippets from the tool. Do NOT answer, translate, or summarize for the end-user.""",
    ),
    ("human", "{input}"),
])

def run_agent1(user_query: str) -> str:
    chain = agent1_prompt | llm_with_tools
    response = chain.invoke({"input": user_query})

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        keyword = tool_call["args"].get("search_keyword", user_query)
        print(f"   [Agent 1 calling Tool with keyword: '{keyword}']")
        return search_knowledge_base.invoke(keyword)

    return search_knowledge_base.invoke(user_query)


# ==============================================================================
# 3. AGENT 2: Report Generator Agent (Strict Single Language Enforcement)
# ==============================================================================
agent2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Agent 2 (Report Generator Agent).
Your role is to synthesize retrieved English snippets into a highly structured, clean, professional response.

CONVERSATION HISTORY:
{chat_history}

RETRIEVED SNIPPETS FROM AGENT 1:
{retrieved_snippets}

CRITICAL STRICT LANGUAGE RULE (ZERO TOLERANCE FOR MIXED LANGUAGES):
1. DETERMINE USER LANGUAGE:
   - Examine the user's latest message ("{input}").
   - If the user asked in ENGLISH -> The ENTIRE response (headings, tables, content, disclaimer, AND follow-up question) MUST BE 100% IN ENGLISH.

2. DO NOT MIX LANGUAGES UNDER ANY CIRCUMSTANCES:
   - NEVER end an English response with a Thai follow-up question or vice versa.
   - Ensure the closing sentence matches the input language strictly.

STRICT GROUNDING & ANTI-HALLUCINATION RULES:
1. STRICT BOUNDARY: Base your response ONLY on the provided RETRIEVED SNIPPETS. Do NOT introduce external knowledge, unmentioned company policies, or unrelated topics (e.g., do NOT mention resignation, leave policies, or general benefits unless explicitly in the snippets).
2. TOPIC FOCUS: Address ONLY the user's specific query ("{input}"). If the snippets do not contain information to fully answer the query, explicitly state what is missing instead of guessing.
3. FOLLOW-UP QUESTION RULE: 
   - The follow-up question MUST be strictly related to the CURRENT TOPIC ("{input}").
   - Example for Travel Allowance: "Would you like to know more about the meal allowance daily rates or expense claim procedure?"
   - Do NOT ask about unrelated company policies.

FORMATTING & STYLE RULES:
1. NO EMOJIS: Do NOT use any emojis or icons anywhere.
2. USE TABLES FOR COMPARISONS/RULES: Always present policies, duration, numbers, or allowances using Markdown Tables.
3. STRUCTURE: Use Markdown headers ('##', '###'), bold text, and concise bullet points.
4. DIRECTNESS: Be direct, objective, and clear. Avoid fluff.

NO INFO FALLBACK:
- If English user input: "I'm sorry, but this specific information is not available in the knowledge base."


FOLLOW-UP QUESTION RULE:
- End with EXACTLY ONE follow-up question.
- MUST be written in the SAME language as the rest of your response.
  - English Example: "Would you like to know more about the resignation process or HR guidelines?"
  """,
    ),
    ("human", "{input}"),
])

agent2_chain = agent2_prompt | llm


# ==============================================================================
# PIPELINE ORCHESTRATION WITH GRADIO STREAMING
# ==============================================================================
def chat_pipeline(message: str, history: list):
    formatted_history = ""
    recent_history = history[-4:] if len(history) > 4 else history
    for item in recent_history:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            formatted_history += f"User: {item[0]}\nAssistant: {item[1]}\n"
        elif isinstance(item, dict):
            formatted_history += (
                f"{item.get('role', 'User')}: {item.get('content', '')}\n"
            )

    try:
        # Step 0: Safety Guardrail
        print("\n>>>>> Step 0: Checking Input Safety with Prompt Guard...")
        if not check_prompt_safety(message):
            yield "Security Alert: Flagged unsafe input or prohibited command detected. Please rephrase your query."
            return

        # Step 1: Agent 1 Execution
        print(
            ">>>>> Step 1: Executing Agent 1 (Data Retriever with Tool"
            " Calling)..."
        )
        snippets = run_agent1(message)

        # Step 2: Agent 2 Streaming Execution
        print(
            ">>>>> Step 2: Executing Agent 2 (Report Generator Streaming)..."
        )
        partial_response = ""
        for chunk in agent2_chain.stream({
            "input": message,
            "chat_history": formatted_history,
            "retrieved_snippets": snippets,
        }):
            partial_response += chunk.content
            yield partial_response

    except Exception as e:
        yield f"An error occurred: {str(e)}"


# ==============================================================================
# GRADIO UI
# ==============================================================================
def create_ui():
    demo = gr.ChatInterface(
        fn=chat_pipeline,
        title="BBL Multi-Agent Enterprise RAG",
       
    )
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue().launch(server_name="127.0.0.1", share=False)