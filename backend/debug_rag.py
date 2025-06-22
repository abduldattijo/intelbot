# debug_rag.py - Debug script to test RAG system

import os
from dotenv import load_dotenv
import json

load_dotenv()


def debug_rag_system():
    """Debug the RAG system step by step"""

    print("🔍 RAG System Debug")
    print("=" * 50)

    try:
        # Import after loading env
        from rag_system import get_rag_system

        # Initialize RAG system
        rag_system = get_rag_system()
        print("✅ RAG system initialized")

        # Check current stats
        stats = rag_system.get_index_stats()
        print(f"📊 Current RAG stats: {json.dumps(stats, indent=2)}")

        # Add a test document
        test_content = """
        SECURITY ALERT - LAGOS STATE

        Date: June 22, 2025
        Classification: RESTRICTED

        INCIDENT SUMMARY:
        Armed robbery incident reported in Victoria Island area of Lagos State.
        Three suspects armed with AK-47 rifles targeted a commercial bank.
        No casualties reported. Suspects fled towards Ikoyi area.

        THREAT ASSESSMENT:
        - Threat Level: HIGH
        - Location: Victoria Island, Lagos
        - Weapons: AK-47 rifles, pistols
        - Suspects: 3 individuals
        - Escape route: Towards Ikoyi

        RECOMMENDATIONS:
        - Increase patrols in Victoria Island and Ikoyi areas
        - Alert all banks in Lagos Island area
        - Deploy rapid response teams
        """

        print("\n📄 Adding test document...")
        chunks_added = rag_system.add_document(
            doc_id="debug_test_001",
            filename="test_security_alert.txt",
            content=test_content,
            metadata={"test": True, "location": "Lagos"}
        )
        print(f"✅ Added {chunks_added} chunks to index")

        # Check stats again
        stats_after = rag_system.get_index_stats()
        print(f"📊 Stats after adding document: {json.dumps(stats_after, indent=2)}")

        # Test retrieval
        print("\n🔎 Testing retrieval...")
        test_query = "What happened in Lagos?"
        chunks = rag_system.retrieve_relevant_chunks(test_query, k=3)
        print(f"📋 Retrieved {len(chunks)} chunks for query: '{test_query}'")

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i + 1}:")
            print(f"    Similarity: {chunk['similarity']:.4f}")
            print(f"    Filename: {chunk['filename']}")
            print(f"    Text preview: {chunk['text'][:100]}...")
            print()

        # Test full query
        print("🤖 Testing full query...")
        result = rag_system.query(test_query, k=3)
        print(f"📝 Query result keys: {list(result.keys())}")
        print(f"🎯 Context chunks: {result.get('context_chunks', 'N/A')}")
        print(f"📚 Sources: {len(result.get('sources', []))}")
        print(f"❓ No results: {result.get('no_results', False)}")

        if result.get('response'):
            print(f"📄 Response preview: {result['response'][:200]}...")

        # Test database contents
        print("\n💾 Testing database...")
        import sqlite3
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 Documents in database: {doc_count}")

        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"🧩 Chunks in database: {chunk_count}")

        if chunk_count > 0:
            cursor.execute("SELECT filename, chunk_index, LENGTH(chunk_text) FROM document_chunks LIMIT 3")
            sample_chunks = cursor.fetchall()
            print("📋 Sample chunks:")
            for filename, idx, length in sample_chunks:
                print(f"  - {filename}, chunk {idx}, {length} chars")

        conn.close()

        # Test FAISS index
        print(f"\n🗂️ FAISS Index:")
        print(f"  Total vectors: {rag_system.index.ntotal}")
        print(f"  Dimension: {rag_system.index.d}")
        print(f"  Index type: {type(rag_system.index)}")
        print(f"  Metadata entries: {len(rag_system.index_metadata)}")

        # Test embeddings
        print(f"\n🧠 Testing embeddings...")
        test_text = "security threat Lagos"
        embedding = rag_system.embedding_model.encode([test_text])
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {(embedding ** 2).sum() ** 0.5}")

    except Exception as e:
        print(f"❌ Error during debug: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_rag_system()