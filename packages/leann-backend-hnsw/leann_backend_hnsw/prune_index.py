import os
import struct
from pathlib import Path

from .convert_to_csr import (
    EXPECTED_HNSW_FOURCCS,
    NULL_INDEX_FOURCC,
    read_struct,
    read_vector_raw,
)


def _write_vector_raw(f_out, count: int, data_bytes: bytes) -> None:
    """Write a vector in the same binary layout as read_vector_raw reads: <Q count> + raw bytes."""
    f_out.write(struct.pack("<Q", count))
    if count > 0 and data_bytes:
        f_out.write(data_bytes)


def prune_embeddings_preserve_graph(input_filename: str, output_filename: str) -> bool:
    """
    Copy an original (non-compact) HNSW index file while pruning the trailing embedding storage.
    Preserves the graph structure and metadata exactly; only writes a NULL storage marker instead of
    the original storage fourcc and payload.

    Returns True on success.
    """
    print(f"Pruning embeddings from {input_filename} to {output_filename}")
    print("--------------------------------")
    # running in mode is-recompute=True and is-compact=False
    in_path = Path(input_filename)
    out_path = Path(output_filename)

    try:
        with open(in_path, "rb") as f_in, open(out_path, "wb") as f_out:
            # Header
            index_fourcc = read_struct(f_in, "<I")
            if index_fourcc not in EXPECTED_HNSW_FOURCCS:
                # Still proceed, but this is unexpected
                pass
            f_out.write(struct.pack("<I", index_fourcc))

            d = read_struct(f_in, "<i")
            ntotal_hdr = read_struct(f_in, "<q")
            dummy1 = read_struct(f_in, "<q")
            dummy2 = read_struct(f_in, "<q")
            is_trained = read_struct(f_in, "?")
            metric_type = read_struct(f_in, "<i")
            f_out.write(struct.pack("<i", d))
            f_out.write(struct.pack("<q", ntotal_hdr))
            f_out.write(struct.pack("<q", dummy1))
            f_out.write(struct.pack("<q", dummy2))
            f_out.write(struct.pack("<?", is_trained))
            f_out.write(struct.pack("<i", metric_type))

            if metric_type > 1:
                metric_arg = read_struct(f_in, "<f")
                f_out.write(struct.pack("<f", metric_arg))

            # Vectors: assign_probas (double), cum_nneighbor_per_level (int32), levels (int32)
            cnt, data = read_vector_raw(f_in, "d")
            _write_vector_raw(f_out, cnt, data)

            cnt, data = read_vector_raw(f_in, "i")
            _write_vector_raw(f_out, cnt, data)

            cnt, data = read_vector_raw(f_in, "i")
            _write_vector_raw(f_out, cnt, data)

            # Probe potential extra alignment/flag byte present in some original formats
            probe = f_in.read(1)
            if probe:
                if probe == b"\x00":
                    # Preserve this unexpected 0x00 byte
                    f_out.write(probe)
                else:
                    # Likely part of the next vector; rewind
                    f_in.seek(-1, os.SEEK_CUR)

            # Offsets (uint64) and neighbors (int32)
            cnt, data = read_vector_raw(f_in, "Q")
            _write_vector_raw(f_out, cnt, data)

            cnt, data = read_vector_raw(f_in, "i")
            _write_vector_raw(f_out, cnt, data)

            # Scalar params
            entry_point = read_struct(f_in, "<i")
            max_level = read_struct(f_in, "<i")
            ef_construction = read_struct(f_in, "<i")
            ef_search = read_struct(f_in, "<i")
            dummy_upper_beam = read_struct(f_in, "<i")
            f_out.write(struct.pack("<i", entry_point))
            f_out.write(struct.pack("<i", max_level))
            f_out.write(struct.pack("<i", ef_construction))
            f_out.write(struct.pack("<i", ef_search))
            f_out.write(struct.pack("<i", dummy_upper_beam))

            # Storage fourcc (if present) — write NULL marker and drop any remaining data
            try:
                read_struct(f_in, "<I")
                # Regardless of original, write NULL
                f_out.write(struct.pack("<I", NULL_INDEX_FOURCC))
                # Discard the rest of the file (embedding payload)
                # (Do not copy anything else)
            except EOFError:
                # No storage section; nothing else to write
                pass

        return True
    except Exception:
        # Best-effort cleanup
        try:
            if out_path.exists():
                out_path.unlink()
        except OSError:
            pass
        return False


def prune_embeddings_preserve_graph_inplace(index_file_path: str) -> bool:
    """
    Convenience wrapper: write pruned file to a temporary path next to the
    original, then atomically replace on success.
    """
    print(f"Pruning embeddings from {index_file_path} to {index_file_path}")
    print("--------------------------------")
    # running in mode is-recompute=True and is-compact=False
    src = Path(index_file_path)
    tmp = src.with_suffix(".pruned.tmp")
    ok = prune_embeddings_preserve_graph(str(src), str(tmp))
    if not ok:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False
    try:
        os.replace(str(tmp), str(src))
    except Exception:
        # Rollback on failure
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False
    return True
