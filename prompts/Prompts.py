from typing import Dict, List

from Models.models import messageSchema

SYSTEM_MESSAGE = (
    '''
    you are a RAG QA assistant.

    you MUST follow these rules:
    1) answer using ONLY the provided CONTEXT.
    2) If the CONTEXT does not contain enough information to answer, output:
    {"answer":"I don't know","confidence":0.3,"citations":[]}
    3) DO NOT use outside knowledge.
    4) DO NOT guess.
    5) Output MUST be valid JSON only (no extra text).
    6) citations must be a list of { "doc_id": "...", "chunk_id": <int> } for the chunks you used.
    7) confidence must be between 0 and 1.
    8) evidence must be a list of 1 to 3 short direct quotes (10 to 30 words each)
    10) If you cannot provide evidence quotes, you MUST refuse:
    {"answer":"I don't know","confidence":0.3,"citations":[],"evidence":[]}

    '''
)
def get_messages(state : messageSchema) -> List[Dict]:
    question = state.user_question
    context_chunks = state.context_chunks
    
    context_block_lines = []
    for c in context_chunks:
        context_block_lines.append(
            f"[SOURCE doc={c['doc_id']} chunk={c['chunk_id']}]\n{c['text']}\n"
        )
    context_block = "\n".join(context_block_lines).strip()
    user_message = (
        f"""
        <context>
        {context_block}
        </context>
        Question:{question}
        """
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE.strip()},
        {"role": "user", "content": user_message.strip()},
    ]
