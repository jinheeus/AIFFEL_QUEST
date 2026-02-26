import pandas as pd
from langchain_openai import ChatOpenAI  # Keep for type hinting or fallback if needed
from common.model_factory import ModelFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm


def evaluate_agentic_metrics(input_path, output_path):
    print(f"Loading results for Agentic Evaluation from {input_path}...")
    df = pd.read_csv(input_path)

    # LLM Judge (Configurable)
    llm = ModelFactory.get_eval_model(level="light", temperature=0.0)

    routing_scores = []
    persona_scores = []
    topic_scores = []
    proc_scores = []

    print("Starting Agentic Evaluation (LLM-as-a-Judge)...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row.get("question", "")
        answer = row.get("answer", "")
        category = row.get("category", "")
        persona = row.get("persona", "")  # Expecting 'common', 'auditor', 'manager'

        # 1. Routing Accuracy Evaluation
        routing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator for a RAG system Router.
Determine if the chosen 'Category' is appropriate for the given 'Question'.

Categories:
- 'search': General questions about regulations, cases, standards.
- 'stats': Questions asking for statistics, counts, top-k, sums, or trends.
- 'compare': Questions asking to compare two or more entities.

Output '1' if the category is appropriate, '0' if not. Only output the number.
""",
                ),
                ("human", "Question: {question}\nChosen Category: {category}"),
            ]
        )

        try:
            routing_chain = routing_prompt | llm | StrOutputParser()
            routing_score = routing_chain.invoke(
                {"question": question, "category": category}
            )
            routing_scores.append(int(routing_score.strip()))
        except:
            routing_scores.append(0)

        # 2. Persona Adherence Evaluation
        # [Refinement] Skip Judgment queries as they follow SOP/Adversarial flow
        if category == "judgment":
            persona_scores.append(-1)  # Mark as N/A
        else:
            persona_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
    You are an expert evaluator for a RAG system Persona.
    Determine if the 'Answer' style and format matches the chosen 'Persona'.

    Personas Criteria:
    - 'common': Friendly or Formal-Polite. Uses polite endings (~해요, ~합니다, ~입니다). Explanatory and easy to understand for the general public.
    - 'auditor': **Structured**. MUST include the 3 core parts: 1. Violation (위반 사항), 2. Basis (판단 근거), 3. Action/Measure (조치 사항). Tone: Dry, factual (~함, ~임).
    - 'manager': **Executive Briefing**. MUST include the 3 core parts: 1. Status (현황/행동), 2. Risk (문제점/리스크), 3. Insight (시사점/대책). Tone: Concise, bullet points.

    Output '1' if the answer follows the specific format and tone for the persona, '0' if not. Only output the number.
    """,
                    ),
                    ("human", "Persona: {persona}\nAnswer: {answer}"),
                ]
            )
            try:
                persona_chain = persona_prompt | llm | StrOutputParser()
                persona_score = persona_chain.invoke(
                    {"persona": persona, "answer": answer}
                )
                persona_scores.append(int(persona_score.strip()))
            except:
                persona_scores.append(0)

        # 3. Audit Topic Match (Retrieval Quality)
        topic_score = 0
        contexts = row.get("contexts", [])
        if contexts and isinstance(contexts, str):
            import ast

            try:
                contexts_list = ast.literal_eval(contexts)
            except:
                contexts_list = [contexts]
        elif isinstance(contexts, list):
            contexts_list = contexts
        else:
            contexts_list = []

        context_text = "\n".join([str(c) for c in contexts_list])

        topic_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator for an Audit RAG system Retriever.
Rate the Relevance of the 'Retrieved Context' to the 'Audit Query' on a scale of 1 to 5.

Evaluation Criteria:
5 (Perfect): The context contains specific audit cases, regulations, or statistics that perfectly match the core topic and intent of the query.
4 (Good): The context is highly relevant but may include some diverse or slightly tangential cases.
3 (Fair): The context is generally related to the topic (e.g., same category) but lacks specific direct matches for the detailed query.
2 (Weak): The context mentions keywords but focuses on different aspects or unrelated cases.
1 (Irrelevant): The context has no semantic relation to the query.

Output only the number (1-5).
""",
                ),
                ("human", "Audit Query: {question}\nRetrieved Context: {context_text}"),
            ]
        )

        try:
            topic_chain = topic_prompt | llm | StrOutputParser()
            t_score = topic_chain.invoke(
                {"question": question, "context_text": context_text[:15000]}
            )
            topic_score = int(t_score.strip())
        except:
            topic_score = 1

        topic_scores.append(topic_score)

        # 4. Procedural Completeness
        proc_score = 0
        proc_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator for an Audit RAG system.
Rate the 'Procedural Completeness' of the Retrieved Context for answering the Query on a scale of 1 to 5.
Focus on whether the context provides the necessary *procedures*, *criteria*, or *actionable steps* asked for.

Evaluation Criteria:
5 (Complete): The context contains clear, detailed procedures, judgment criteria, or regulation clauses sufficient to fully answer "How" or "What" questions.
4 (Sufficient): The context contains enough info to form a good answer, though some minor details might be inferred.
3 (Partial): The context provides some clues or related standards but lacks a direct answer key.
2 (Insufficient): The context is vague or only tangentially related to the procedure asked.
1 (None): The context contains no procedural information relevant to the question.

Output only the number (1-5).
""",
                ),
                ("human", "Audit Query: {question}\nRetrieved Context: {context_text}"),
            ]
        )

        try:
            proc_chain = proc_prompt | llm | StrOutputParser()
            p_res = proc_chain.invoke(
                {"question": question, "context_text": context_text[:15000]}
            )
            proc_score = int(p_res.strip())
        except:
            proc_score = 1

        proc_scores.append(proc_score)

        # 5. KPR (Key Point Recall) - Proxy
        # Strategy: Extract 3 key bullet points from Context, then check if Answer contains them.
        kpr_score = 0.0
        kpr_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator. Use the following steps to evaluate 'Key Point Recall':

1. Read the [Context] and extract exactly 3 distinct, specific key facts or procedural steps (Key Points) that are most relevant to the [Question].
2. Read the [Answer].
3. For each Key Point, determine if it is clearly mentioned in the [Answer].
4. Calculate the percentage of Key Points found (e.g., 0/3=0.0, 1/3=0.33, 2/3=0.66, 3/3=1.0).

Output ONLY the floating point number (0.0 to 1.0).
""",
                ),
                (
                    "human",
                    "Question: {question}\nContext: {context_text}\nAnswer: {answer}",
                ),
            ]
        )

        try:
            kpr_chain = kpr_prompt | llm | StrOutputParser()
            kpr_res = kpr_chain.invoke(
                {
                    "question": question,
                    "context_text": context_text[:10000],
                    "answer": answer,
                }
            )
            import re

            match = re.search(r"0\.\d+|1\.0|0|1", kpr_res)
            if match:
                kpr_score = float(match.group())
            else:
                kpr_score = 0.0
        except:
            kpr_score = 0.0

        # Store temporarily in a list if you want, or just append to DF later
        # We need a list to store it for the loop
        if "kpr_scores" not in locals():
            kpr_scores = []
        kpr_scores.append(kpr_score)

    df["routing_accuracy"] = routing_scores
    df["persona_adherence"] = persona_scores
    df["audit_topic_match"] = topic_scores
    df["kpr_score"] = kpr_scores

    # Calculate Averages
    avg_routing = df["routing_accuracy"].mean()
    df["persona_adherence"] = persona_scores

    # Filter out N/A (-1) for Persona Average
    valid_persona_scores = [s for s in persona_scores if s != -1]
    avg_persona = (
        sum(valid_persona_scores) / len(valid_persona_scores)
        if valid_persona_scores
        else 0.0
    )
    avg_topic = df["audit_topic_match"].mean()
    avg_proc = df["procedural_completeness"].mean()
    avg_kpr = df["kpr_score"].mean()

    print(f"\n[Agentic Evaluation Results]")
    print(f"- Routing Accuracy: {avg_routing:.2f}")
    print(f"- Persona Adherence: {avg_persona:.2f}")
    print(f"- Audit Topic Match (1-5): {avg_topic:.2f}")
    print(f"- Procedural Completeness (1-5): {avg_proc:.2f}")
    print(f"- Key Point Recall (KPR): {avg_kpr:.2f}")

    print(f"Saving evaluated results to {output_path}...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Done!")
