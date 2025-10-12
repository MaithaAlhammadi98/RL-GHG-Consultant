#!/usr/bin/env python3
"""
Quick script to populate the Chroma database with GHG documents
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from backend.embedding_generation import Embedding_Generation
from backend.rag_process import rag_process

def main():
    print(" Starting database population...")
    
    # Initialize embedding generation
    eg = Embedding_Generation()
    
    # Check if database is empty
    existing = eg.collection.get(limit=1)
    if existing['documents']:
        print(f" Database already has {len(existing['documents'])} documents")
        response = input("Clear existing data and repopulate? (y/N): ")
        if response.lower() != 'y':
            print(" Keeping existing data")
            return
    
    print("️  Clearing existing data...")
    eg.collection.delete()
    
    print(" Processing PDF files...")
    data_dir = Path("src/data")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    print("\n Processing files...")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
        try:
            # Process the PDF
            eg.process_pdf(str(pdf_file))
            print(f"   Success")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Check final count
    final = eg.collection.get()
    total_docs = len(final['documents']) if final['documents'] else 0
    print(f"\n Database populated with {total_docs} documents!")
    
    # Test a query
    print("\n Testing query...")
    rag = rag_process()
    chunks, metas = rag.query_documents("What is GHG?", n_results=3)
    print(f"Test query returned {len(chunks)} chunks")
    if chunks:
        print(f"Sample chunk: {chunks[0][:100]}...")

if __name__ == "__main__":
    main()
