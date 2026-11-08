"""Validates Engram metadata against the checkpoint reference:

Engram hashing: multipliers int64 [2, 4], primes int64
                 [2, 3, 8], offsets int64 [2, 24], the compressed pad id int32,
                 then the compressed token map: count int32, ids int32 [count].
                 Computed by the checkpoint's own inference/engram.py.

usage: python tests/reference/engram.py CHECKPOINT_DIR [--output FILE]
"""

import importlib.util
import argparse
import io
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def engram(checkpoint, out):
    spec = importlib.util.spec_from_file_location("engram", checkpoint / "inference" / "engram.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((checkpoint / "inference" / "config.json").read_text())
    args = types.SimpleNamespace(**config, max_batch_size=1, max_seq_len=1)
    backend = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))

    class Shim:  # what build_compressed_token_map reads of a transformers tokenizer
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    layout = module.EngramLayout.from_args(args)
    state = module.NgramHashState(args, layout, Shim())
    tables = [state.multipliers, state.primes, state.offsets]
    with io.BytesIO() as f:
        for t in tables:
            f.write(np.ascontiguousarray(t.numpy().astype("<i8")).tobytes())
        token_map = state.token_map.numpy().astype("<i4")
        f.write(struct.pack("<ii", int(state.pad_id), len(token_map)))
        f.write(token_map.tobytes())
        result = f.getvalue()
    if out:
        Path(out).write_bytes(result)
    digest = 1469598103934665603
    for byte in result:
        digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    print(f"{len(result)} reference bytes; fingerprint {digest:016x}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Engram metadata with the checkpoint's reference implementation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", help="optional output file; otherwise print only the reference fingerprint")
    args = parser.parse_args()
    engram(args.checkpoint, args.output)
