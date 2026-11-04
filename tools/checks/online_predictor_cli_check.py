"""Check predictor option combinations without loading a model or starting Metal."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "build/myds4"
    missing = root / "build/.online-predictor-missing-checkpoint"
    if missing.exists():
        raise RuntimeError("the CLI check requires its checkpoint path to be absent")
    base = [str(binary), "-m", str(missing), "-p", "x"]
    clean = dict(os.environ)
    clean.pop("MYDS4_PROFILE", None)
    clean.pop("MYDS4_API_KEY", None)
    missing_tokenizer = f"cannot open {missing}/tokenizer.json"
    # Tokenizer failure proves that ordinary argument validation completed.
    # The benchmark case ends at a later validation, before it could
    # initialize Metal.
    cases = []

    def add(name, args=(), profile=None, expected=missing_tokenizer, online=True):
        flags = ["--online-predictor"] if online else []
        cases.append((name, flags + list(args), profile, expected))

    add("ordinary build")
    add("profiling environment empty", profile="")
    add("profiling environment nonempty", profile="1")
    add("persistent state alone", ["--predictor-state", "unused-state.bin"], online=False)
    add("persistent state with explicit enable", ["--predictor-state", "unused-state.bin"])
    add("empty state path", ["--predictor-state", ""], expected="needs a nonempty path")
    add("removed fitted predictor", ["--predictor", "unused"], expected="usage:", online=False)
    add("removed fitting trace", ["--fit-fd", "3"], expected="usage:", online=False)
    add("removed numeric rank", ["--online-predictor", "64"], expected="usage:", online=False)
    add("removed server", ["--serve", "1234"], expected="usage:")
    add("removed live view", ["--tui"], expected="usage:")
    add("removed prompt rendering", ["--render", "case.json"], expected="usage:", online=False)
    for args in (
        ["--substitute", "1"],
        ["--gpu-route"],
        ["--routes-fd", "3"],
        ["--dump-fd", "3"],
        ["--check-prefill"],
        ["--check-batch"],
        ["--perplexity"],
        ["--image", "unused"],
        ["--vision-fd", "3"],
        ["--experts", "unused"],
        [
            "--draft",
            "2",
            "--draft-cache",
            "32",
            "--draft-trim",
            "--cache-half-life",
            "10",
            "--cache-read-cost",
            "--substitute",
            "0.5",
            "--gpu-route",
        ],
    ):
        add(args[0], args)
    for name, args, profile, expected in (
        (
            "benchmark argument",
            ["--bench", "--draft-cache", "32"],
            None,
            "--draft-cache needs --draft",
        ),
        (
            "image generation constraint",
            ["--image", "unused", "--tokens", "1"],
            None,
            "--image works",
        ),
        ("sampled drafts", ["--draft", "2", "-t", "1"], None, missing_tokenizer),
        ("context lookup", ["--lookup", "15", "-t", "1", "--top-p", "1"], None, missing_tokenizer),
        ("context lookup off", ["--lookup", "off"], None, missing_tokenizer),
        ("context lookup with DSpark", ["--lookup", "15", "--draft", "2", "--draft-cache", "32"], None, missing_tokenizer),
        ("context lookup with shared experts", ["--lookup", "15", "--expert-batch"], None, missing_tokenizer),
        ("dependency-pruned prefill", ["--prefill-prune", "--ced", "off"], None, missing_tokenizer),
        ("pruned prefill with lookup", ["--prefill-prune", "--lookup", "15", "--expert-batch"], None, missing_tokenizer),
        ("invalid context lookup", ["--lookup", "16"], None, "--lookup"),
        (
            "temperature and nucleus sampling",
            ["--draft", "2", "-t", "1", "--top-p", "1", "--seed", "42"],
            None,
            missing_tokenizer,
        ),
        ("negative temperature", ["-t", "-1"], None, "temperature"),
        ("nonfinite temperature", ["-t", "nan"], None, "temperature"),
        ("invalid top-p", ["--top-p", "1.1"], None, "top-p"),
        ("nonfinite top-p", ["--top-p", "nan"], None, "top-p"),
        ("negative seed", ["--seed", "-1"], None, "seed"),
        ("invalid thinking effort", ["--think", "101"], None, "--think"),
        (
            "draft cache constraint",
            ["--draft-cache", "32"],
            None,
            "--draft-cache needs --draft",
        ),
        ("draft trim constraint", ["--draft-trim"], None, "--draft-trim needs --draft"),
        ("memory reserve", ["--reserve", "4"], None, missing_tokenizer),
        ("fractional memory reserve", ["--reserve", "2.5"], None, missing_tokenizer),
        ("zero memory reserve", ["--reserve", "0"], None, "--reserve takes 2 to 1024 GiB"),
        ("small memory reserve", ["--reserve", "1"], None, "--reserve takes 2 to 1024 GiB"),
        ("large memory reserve", ["--reserve", "2000"], None, "--reserve takes 2 to 1024 GiB"),
        ("nonnumeric memory reserve", ["--reserve", "lots"], None, "--reserve takes 2 to 1024 GiB"),
    ):
        add(name, args, profile, expected)
    failures = []
    for name, args, profile, expected in cases:
        env = clean.copy()
        if profile is not None:
            env["MYDS4_PROFILE"] = profile
        result = subprocess.run(base + args, env=env, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 or expected not in result.stderr:
            failures.append(name)
            first_line = result.stderr.splitlines()[0] if result.stderr else "<no error>"
            print(f"FAIL {name}: expected {expected!r}; got {first_line!r}", file=sys.stderr)
    print(f"online predictor CLI: {len(cases) - len(failures)}/{len(cases)} checks passed")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
