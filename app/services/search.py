import logging

import openai
from sqlalchemy.orm import Session

from app.services.openai_client import create_openai_client
from app.services import chunk_store

logger = logging.getLogger(__name__)

#TODO: we will want to add configuration to point pastors if too deep
#TODO: we will want to query our previous answers to avoid extra API calls if the same question is asked multiple times
COMPLETION_MODEL = "gpt-4o"
SYSTEM_PROMPT = """You are a Christian theological assistant helping users understand the Bible through the provided sources.

Answer questions using ONLY the document excerpts provided. Do not draw on outside knowledge.
- Cite the source title without the chunk index for every claim
- If the provided excerpts do not contain enough information to answer, say so clearly — do not speculate or fill in gaps
- Present differing interpretations fairly if they appear in the source material

Format your response using Markdown: use **bold** for emphasis, headings (##, ###) to organise longer answers, and bullet lists where appropriate.
Write with clarity and pastoral warmth, grounded entirely in the provided documents."""


def answer_question(
    query: str,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    api_key: str | None = None,
    db: Session | None = None,
) -> dict:
    client = create_openai_client(api_key)

    try:
        embed_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        )
        query_embedding = embed_response.data[0].embedding

        hits = chunk_store.search_chunks(
            query_vector=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
            db=db,
        )

        sources = []
        context_parts = []
        for hit in hits:
            sources.append(
                {
                    "document_id": hit.document_id,
                    "title": hit.title,
                    "author": hit.author,
                    "chunk_index": hit.chunk_index,
                    "content": hit.content,
                    "score": hit.score,
                }
            )
            context_parts.append(
                f"[Source: {hit.title}, chunk {hit.chunk_index}]\n{hit.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Question: {query}\n\nDocument excerpts:\n\n{context}"

        logger.info("Generating answer for query: %s", query[:80])

        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0,
        )

        answer = completion.choices[0].message.content
        return {"answer": answer, "sources": sources}
    finally:
        client.close()
