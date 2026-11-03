"""Generates the tables myds4 embeds at build time, from the checkpoint:

data/engram.bin  Engram hashing: multipliers int64 [2, 4], primes int64
                 [2, 3, 8], offsets int64 [2, 24], the compressed pad id int32,
                 then the compressed token map: count int32, ids int32 [count].
                 Computed by the checkpoint's own inference/engram.py.
data/unicode.bin Unicode general-category classes for the pre-tokenizer:
                 uint32 runs, (first code point << 3) | class, with classes
                 0 other, 1 letter, 2 mark, 3 number, 4 punctuation, 5 symbol.

usage: python tools/data/gen_data.py CHECKPOINT_DIR
"""

import importlib.util
import json
import struct
import sys
import types
import unicodedata
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
    with open(out, "wb") as f:
        for t in tables:
            f.write(np.ascontiguousarray(t.numpy().astype("<i8")).tobytes())
        token_map = state.token_map.numpy().astype("<i4")
        f.write(struct.pack("<ii", int(state.pad_id), len(token_map)))
        f.write(token_map.tobytes())
    print(f"{out}: {len(token_map)} tokens, multipliers {state.multipliers.tolist()}")


def unicode(out):
    classes = {"L": 1, "M": 2, "N": 3, "P": 4, "S": 5}
    runs, last = [], -1
    for c in range(0x110000):
        k = classes.get(unicodedata.category(chr(c))[0], 0)
        if k != last:
            runs.append(c << 3 | k)
            last = k
    Path(out).write_bytes(struct.pack(f"<{len(runs)}I", *runs))
    print(f"{out}: {len(runs)} runs (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    checkpoint = Path(sys.argv[1])
    data = Path(__file__).resolve().parents[2] / "data"
    data.mkdir(exist_ok=True)
    engram(checkpoint, data / "engram.bin")
    unicode(data / "unicode.bin")
