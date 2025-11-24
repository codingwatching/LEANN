#!/usr/bin/env python3
"""
Test script to reproduce issue #159: Slow search performance
Configuration:
- GPU: 4090×1
- embedding_model: BAAI/bge-large-zh-v1.5
- data size: 180M text (~90K chunks)
- beam_width: 10 (though this is mainly for DiskANN, not HNSW)
- backend: hnsw
"""

import os
import time
from pathlib import Path

from leann.api import LeannBuilder, LeannSearcher, SearchResult

os.environ["LEANN_LOG_LEVEL"] = "DEBUG"

# Configuration matching the issue
INDEX_PATH = "./test_issue_159.leann"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
BACKEND_NAME = "hnsw"


def generate_test_data(num_chunks=90000, chunk_size=2000):
    """Generate test data similar to 180MB text (~90K chunks)"""
    # Each chunk is approximately 2000 characters
    # 90K chunks * 2000 chars ≈ 180MB
    chunks = []
    base_text = (
        "这是一个测试文档。LEANN是一个创新的向量数据库，通过图基选择性重计算实现97%的存储节省。"
    )

    for i in range(num_chunks):
        chunk = f"{base_text} 文档编号: {i}. " * (chunk_size // len(base_text) + 1)
        chunks.append(chunk[:chunk_size])

    return chunks


def test_search_performance():
    """Test search performance with different configurations"""
    print("=" * 80)
    print("Testing LEANN Search Performance (Issue #159)")
    print("=" * 80)

    meta_path = Path(f"{INDEX_PATH}.meta.json")
    if meta_path.exists():
        print(f"\n✓ Index already exists at {INDEX_PATH}")
        print("  Skipping build phase. Delete the index to rebuild.")
    else:
        print("\n📦 Building index...")
        print(f"  Backend: {BACKEND_NAME}")
        print(f"  Embedding Model: {EMBEDDING_MODEL}")
        print("  Generating test data (~90K chunks, ~180MB)...")

        chunks = generate_test_data(num_chunks=90000)
        print(f"  Generated {len(chunks)} chunks")
        print(f"  Total text size: {sum(len(c) for c in chunks) / (1024 * 1024):.2f} MB")

        builder = LeannBuilder(
            backend_name=BACKEND_NAME,
            embedding_model=EMBEDDING_MODEL,
        )

        print("  Adding chunks to builder...")
        start_time = time.time()
        for i, chunk in enumerate(chunks):
            builder.add_text(chunk)
            if (i + 1) % 10000 == 0:
                print(f"    Added {i + 1}/{len(chunks)} chunks...")

        print("  Building index...")
        build_start = time.time()
        builder.build_index(INDEX_PATH)
        build_time = time.time() - build_start
        print(f"  ✓ Index built in {build_time:.2f} seconds")

    # Test search with different complexity values
    print("\n🔍 Testing search performance...")
    searcher = LeannSearcher(INDEX_PATH)

    test_query = "LEANN向量数据库存储优化"

    # Test with default complexity (64)
    print("\n  Test 1: Default complexity (64) `1 ")
    print(f"    Query: '{test_query}'")
    start_time = time.time()
    results: list[SearchResult] = searcher.search(test_query, top_k=10, complexity=64)
    search_time = time.time() - start_time
    print(f"    ✓ Search completed in {search_time:.2f} seconds")
    print(f"    Results: {len(results)} items")

    # Test with default complexity (64)
    print("\n  Test 1: Default complexity (64)")
    print(f"    Query: '{test_query}'")
    start_time = time.time()
    results = searcher.search(test_query, top_k=10, complexity=64)
    search_time = time.time() - start_time
    print(f"    ✓ Search completed in {search_time:.2f} seconds")
    print(f"    Results: {len(results)} items")

    # Test with lower complexity (32)
    print("\n  Test 2: Lower complexity (32)")
    print(f"    Query: '{test_query}'")
    start_time = time.time()
    results = searcher.search(test_query, top_k=10, complexity=32)
    search_time = time.time() - start_time
    print(f"    ✓ Search completed in {search_time:.2f} seconds")
    print(f"    Results: {len(results)} items")

    # Test with even lower complexity (16)
    print("\n  Test 3: Lower complexity (16)")
    print(f"    Query: '{test_query}'")
    start_time = time.time()
    results = searcher.search(test_query, top_k=10, complexity=16)
    search_time = time.time() - start_time
    print(f"    ✓ Search completed in {search_time:.2f} seconds")
    print(f"    Results: {len(results)} items")

    # Test with minimal complexity (8)
    print("\n  Test 4: Minimal complexity (8)")
    print(f"    Query: '{test_query}'")
    start_time = time.time()
    results = searcher.search(test_query, top_k=10, complexity=8)
    search_time = time.time() - start_time
    print(f"    ✓ Search completed in {search_time:.2f} seconds")
    print(f"    Results: {len(results)} items")

    print("\n" + "=" * 80)
    print("Performance Analysis:")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. beam_width parameter is mainly for DiskANN backend, not HNSW")
    print("2. For HNSW, the main parameter affecting search speed is 'complexity'")
    print("3. Lower complexity values (16-32) should provide faster search")
    print("4. The paper mentions ~2 seconds, which likely uses:")
    print("   - Smaller embedding model (~100M params vs 300M for bge-large)")
    print("   - Lower complexity (16-32)")
    print("   - Possibly DiskANN backend for better performance")
    print("\nRecommendations:")
    print("- Try complexity=16 or complexity=32 for faster search")
    print("- Consider using DiskANN backend for better performance on large datasets")
    print("- Or use a smaller embedding model if speed is critical")


if __name__ == "__main__":
    test_search_performance()
