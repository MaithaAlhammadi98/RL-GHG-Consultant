import os
from dotenv import load_dotenv
from groq import AsyncGroq
from backend.embedding_generation import Embedding_Generation
import streamlit as st
import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass
from pathlib import Path

def run_async(coro):
    """Safely run a coroutine in both plain Python and within a running event loop (Jupyter/Streamlit)."""
    try:
        loop = asyncio.get_running_loop() #!!!!!!!alert
    except RuntimeError:
        # No running loop: safe to use run_async
        return run_async(coro)
    else:
        # Running loop exists (Jupyter/Streamlit): run in that loop
        return loop.run_until_complete(coro)

# load project-root .env no matter where this file lives
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
# parents[2] = up from src/backend/<file>.py to project root
# adjust to parents[1] if your file lives only one level below src/

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from backend.embedding_generation import Embedding_Generation

class rag_process:
    def __init__(self):
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
        self.embedding_class = Embedding_Generation()

    # NEW: add metadata_filter param and forward to Chroma via "where"
    def query_documents(
        self,
        question: str,
        n_results: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        query_embedding = self.embedding_class.custom_embeddings([question])

        results = self.embedding_class.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas"],
            where=metadata_filter if metadata_filter else None,  # <-- key line
        )

        # keep your existing return shape
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return docs, metas


    def generate_response(self, question, relevant_chunks, results_metadata):
        # Format context with source information
        formatted_chunks = []
        
        for i, (chunk, metadata) in enumerate(zip(relevant_chunks, results_metadata)):
            # Format with page information if available
            source_info = f"Source: {metadata.get('source', 'Unknown')}"
            if metadata.get('chunk_number'):
                source_info += f" (Chunk {metadata['chunk_number']})"
            
            formatted_chunk = f"{source_info}\n{chunk}"
            formatted_chunks.append(formatted_chunk)
            
        context = "\n\n---\n\n".join(formatted_chunks)  # Added separator for better readability
        
        ghg_assistant = st.session_state.ghg_assistant

        try:
            answer = run_async(ghg_assistant.generate_response(
                user_prompt=question,
                context=context
            ))
        except Exception as e:
            return f"Error generating response: {str(e)}"

        return answer

    def format_context(self, chunks: list, metas: list) -> str:
        """
        Formats the retrieved chunks and metadata into a single string for the LLM context.
        """
        formatted = []
        for ch, md in zip(chunks, metas):
            src = md.get("source", "Unknown")
            pg = md.get("page") or md.get("chunk_number")
            tag = f"{src}" + (f" (p.{pg})" if pg else "")
            formatted.append(f"Source: {tag}\n{ch}")
        return "\n\n---\n\n".join(formatted)
