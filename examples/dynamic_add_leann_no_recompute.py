"""
Dynamic add example for LEANN using HNSW backend without recompute.

- Builds a base index from a directory of documents
- Incrementally adds new documents without recomputing stored embeddings

Defaults:
- Base data: /Users/yichuan/Desktop/code/LEANN/leann/data
- Incremental data: /Users/yichuan/Desktop/code/LEANN/leann/test_add
- Index path: <index_dir>/documents.leann

Usage examples:
  uv run python examples/dynamic_add_leann_no_recompute.py --build-base \
    --base-dir /Users/yichuan/Desktop/code/LEANN/leann/data \
    --index-dir ./test_doc_files

  uv run python examples/dynamic_add_leann_no_recompute.py --add-incremental \
    --add-dir /Users/yichuan/Desktop/code/LEANN/leann/test_add \
    --index-dir ./test_doc_files

Quick recompute test (both true):
  # Recompute build
  uv run python examples/dynamic_add_leann_no_recompute.py --build-base \
    --recompute-build --ef-construction 200 \
    --base-dir /Users/yichuan/Desktop/code/LEANN/leann/data \
    --index-dir ./test_doc_files --index-name documents.leann

  # Recompute add
  uv run python examples/dynamic_add_leann_no_recompute.py --add-incremental \
    --recompute-add --ef-construction 32 \
    --add-dir /Users/yichuan/Desktop/code/LEANN/leann/test_add \
    --index-dir ./test_doc_files --index-name documents.leann
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure we can import from the local packages and apps folders
ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages" / "leann-core" / "src"
HNSW_PKG_DIR = ROOT / "packages" / "leann-backend-hnsw"
APPS_DIR = ROOT / "apps"


# Prefer the installed backend if available (it contains the compiled extension)
def _prefer_installed(pkg_name: str) -> bool:
    try:
        import importlib
        import importlib.util

        spec = importlib.util.find_spec(pkg_name)
        if spec and spec.origin and "site-packages" in spec.origin:
            # ensure the faiss shim/extension is importable from the installed package
            importlib.import_module(f"{pkg_name}.faiss")
            return True
    except Exception:
        pass
    return False


# Prepend paths, but only add the repo backend if the installed one is not present
paths_to_prepend = [CORE_SRC, APPS_DIR]
if not _prefer_installed("leann_backend_hnsw"):
    paths_to_prepend.insert(1, HNSW_PKG_DIR)

for p in paths_to_prepend:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Defer non-stdlib imports until after sys.path setup within functions (avoid E402)


def _load_documents(data_dir: str, required_exts: Optional[list[str]] = None) -> list[Any]:
    from llama_index.core import SimpleDirectoryReader  # type: ignore

    reader_kwargs: dict[str, Any] = {"recursive": True, "encoding": "utf-8"}
    if required_exts:
        reader_kwargs["required_exts"] = required_exts
    documents = SimpleDirectoryReader(data_dir, **reader_kwargs).load_data(show_progress=True)
    return documents


def _ensure_index_dir(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def _index_files(index_path: Path) -> tuple[Path, Path, Path]:
    """Return (passages.jsonl, passages.idx, index.index) paths for a given index base path.

    Note: HNSWBackend writes the FAISS index using the stem (without .leann),
    i.e., for base 'documents.leann' the file is 'documents.index'. We prefer the
    existing file among candidates.
    """
    passages_file = index_path.parent / f"{index_path.name}.passages.jsonl"
    offsets_file = index_path.parent / f"{index_path.name}.passages.idx"
    candidate_name_index = index_path.parent / f"{index_path.name}.index"
    candidate_stem_index = index_path.parent / f"{index_path.stem}.index"
    index_file = candidate_stem_index if candidate_stem_index.exists() else candidate_name_index
    return passages_file, offsets_file, index_file


def _read_meta(index_path: Path) -> dict[str, Any]:
    meta_path = index_path.parent / f"{index_path.name}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _autodetect_index_base(index_dir: Path) -> Optional[Path]:
    """If exactly one *.leann.meta.json exists, return its base path (without .meta.json)."""
    candidates = list(index_dir.glob("*.leann.meta.json"))
    if len(candidates) == 1:
        meta = candidates[0]
        base = meta.with_suffix("")  # remove .json
        base = base.with_suffix("")  # remove .meta
        return base
    return None


def _load_offset_map(offsets_file: Path) -> dict[str, int]:
    if not offsets_file.exists():
        return {}
    with open(offsets_file, "rb") as f:
        return pickle.load(f)


def _next_numeric_id(existing_ids: list[str]) -> int:
    numeric_ids = [int(x) for x in existing_ids if x.isdigit()]
    if not numeric_ids:
        return 0
    return max(numeric_ids) + 1


def build_base_index(
    base_dir: str,
    index_dir: str,
    index_name: str,
    embedding_model: str,
    embedding_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    file_types: Optional[list[str]] = None,
    max_items: int = -1,
    ef_construction: Optional[int] = None,
    recompute_build: bool = False,
) -> str:
    print(f"Building base index from: {base_dir}")
    documents = _load_documents(base_dir, required_exts=file_types)
    if not documents:
        raise ValueError(f"No documents found in base_dir: {base_dir}")

    from chunking import create_text_chunks

    texts = create_text_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_ast_chunking=False,
    )
    if max_items > 0 and len(texts) > max_items:
        texts = texts[:max_items]
        print(f"Limiting to {max_items} chunks")

    index_dir_path = Path(index_dir)
    _ensure_index_dir(index_dir_path)
    index_path = index_dir_path / index_name

    print("Creating HNSW index (non-compact)...")
    from leann.api import LeannBuilder
    from leann.registry import register_project_directory

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=embedding_model,
        embedding_mode=embedding_mode,
        is_recompute=recompute_build,
        is_compact=False,
        efConstruction=(ef_construction if ef_construction is not None else 200),
    )
    for t in texts:
        builder.add_text(t)
    builder.build_index(str(index_path))

    # Register for discovery
    register_project_directory(Path.cwd())

    print(f"Base index built at: {index_path}")
    return str(index_path)


def add_incremental(
    add_dir: str,
    index_dir: str,
    index_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_mode: Optional[str] = None,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    file_types: Optional[list[str]] = None,
    max_items: int = -1,
    ef_construction: Optional[int] = None,
    recompute_add: bool = False,
) -> str:
    print(f"Adding incremental data from: {add_dir}")
    index_dir_path = Path(index_dir)
    index_path = index_dir_path / (index_name or "documents.leann")

    # If specified base doesn't exist, try to auto-detect an existing base
    try:
        _read_meta(index_path)
    except FileNotFoundError:
        auto_base = _autodetect_index_base(index_dir_path)
        if auto_base is not None:
            print(f"Auto-detected index base: {auto_base.name}")
            index_path = auto_base
            _read_meta(index_path)
        else:
            raise FileNotFoundError(
                f"No index metadata found for base '{index_path.name}'. Build base first with --build-base "
                f"or provide --index-name to match an existing index (e.g., 'test_doc_files.leann')."
            )

    # Prepare validated context from core (checks backend/no-recompute and resolves embedding defaults)
    from leann.api import create_incremental_add_context, incremental_add_texts_with_context

    ctx = create_incremental_add_context(
        str(index_path),
        embedding_model=embedding_model,
        embedding_mode=embedding_mode,
        data_dir=add_dir,
        required_exts=file_types,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_items=max_items,
    )

    # Use prepared texts from context to perform the add
    prepared_texts = ctx.prepared_texts or []
    if not prepared_texts:
        print("No new chunks to add.")
        return str(index_path)

    added = incremental_add_texts_with_context(
        ctx,
        prepared_texts,
        ef_construction=ef_construction,
        recompute=recompute_add,
    )

    print(f"Incremental add completed. Added {added} chunks. Index: {index_path}")
    return str(index_path)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic add to LEANN HNSW index without recompute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--build-base", action="store_true", help="Build base index")
    parser.add_argument("--add-incremental", action="store_true", help="Add incremental data")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/Users/yichuan/Desktop/code/LEANN/leann/data",
        help="Base data directory",
    )
    parser.add_argument(
        "--add-dir",
        type=str,
        default="/Users/yichuan/Desktop/code/LEANN/leann/test_add",
        help="Incremental data directory",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./test_doc_files",
        help="Directory containing the index",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="documents.leann",
        help=(
            "Index base file name. If you built via document_rag.py, use 'test_doc_files.leann'. "
            "Default: documents.leann"
        ),
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/contriever",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode",
    )

    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--file-types", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--ef-construction", type=int, default=32)
    parser.add_argument(
        "--recompute-add", action="store_true", help="Enable recompute-mode add (non-compact only)"
    )
    parser.add_argument(
        "--recompute-build",
        action="store_true",
        help="Enable recompute-mode base build (non-compact only)",
    )

    args = parser.parse_args()

    if not args.build_base and not args.add_incremental:
        print("Nothing to do. Use --build-base and/or --add-incremental.")
        return

    index_path_str: Optional[str] = None

    if args.build_base:
        index_path_str = build_base_index(
            base_dir=args.base_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            file_types=args.file_types,
            max_items=args.max_items,
            ef_construction=args.ef_construction,
            recompute_build=args.recompute_build,
        )

    if args.add_incremental:
        index_path_str = add_incremental(
            add_dir=args.add_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            file_types=args.file_types,
            max_items=args.max_items,
            ef_construction=args.ef_construction,
            recompute_add=args.recompute_add,
        )

    # Optional: quick test query using searcher
    if index_path_str:
        try:
            from leann.api import LeannSearcher

            searcher = LeannSearcher(index_path_str)
            query = "what is LEANN?"
            if args.add_incremental:
                query = "what is the multi vector search and how it works?"
            results = searcher.search(query, top_k=5)
            if results:
                print(f"Sample result: {results[0].text[:80]}...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
