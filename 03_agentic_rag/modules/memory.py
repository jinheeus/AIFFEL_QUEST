from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from model_factory import ModelFactory


# --- Summary Prompt ---
SUMMARY_SYSTEM_PROMPT = """
[Role]
You are a Conversation Summarizer for an Audit RAG Chatbot.
Your goal is to maintain a concise running summary of the conversation to preserve context while saving tokens.

[Input]
1. Current Summary: The existing summary of the conversation history.
2. New Lines: The most recent conversation turns that need to be added.

[Output]
Update the 'Current Summary' by incorporating the key information from 'New Lines'.
- Keep the summary concise (under 200 words).
- Preserve important entities (Case names, Regulation Article numbers, Specific user intent).
- If 'New Lines' is empty, return the 'Current Summary' as is.
"""


def summarize_conversation(state: AgentState) -> dict:
    """
    [Node] Summarizes the conversation history to save tokens.
    It takes the *tail* of the conversation (which grew since last summary) and updates the summary.
    """
    print("--- [Node] Summarize Conversation (Memory) ---")

    current_summary = state.get("summary", "")
    messages = state.get("messages", [])

    # 1. Check if summarization is needed
    # Strategy: If history is long (> 6), we summarize everything EXCEPT the last 2 messages (Active turn).
    # This keeps the immediate context fresh while compressing the rest.
    if len(messages) <= 6:
        print(" -> History short, skipping summary.")
        return {"summary": current_summary}

    # 2. Slice messages to summarize
    # Use last 2 for immediate context, summarize the rest.
    # Note: In a real system, we might need a 'last_summarized_index' pointer.
    # Here, we regenerate the summary from scratch or update it?
    # Updating is better. But with 'messages' list being potentially reset by frontend input,
    # we need to be careful.

    # Robust Approach:
    # Summarize ALL messages up to the last 2.
    to_summarize = messages[:-2]

    text_to_summarize = "\n".join(
        [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in to_summarize]
    )

    # 3. Invoke LLM
    llm = ModelFactory.get_eval_model(level="light", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            (
                "human",
                "Current Summary: {current_summary}\n\nNew Lines to Add: {new_lines}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        # Note: If we are re-summarizing the whole 'to_summarize' block every time,
        # we shouldn't pass 'current_summary' recursively unless we are incremental.
        # Since we don't know what was already summarized (without a pointer),
        # let's just Generate a NEW summary for the 'to_summarize' block.
        # This is safer against duplication.

        # Override Prompt for 'Fresh Summary' mode
        fresh_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARY_SYSTEM_PROMPT),
                ("human", "Summarize this conversation:\n{new_lines}"),
            ]
        )

        chain = fresh_prompt | llm | StrOutputParser()
        new_summary = chain.invoke({"new_lines": text_to_summarize})

        print(f" -> Summary Updated: {new_summary[:50]}...")
        return {"summary": new_summary}

    except Exception as e:
        print(f" -> Summary Generation Failed: {e}")
        return {"summary": current_summary}
