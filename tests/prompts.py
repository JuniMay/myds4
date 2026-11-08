"""Check CLI tokenization of rendered server prompts without loading a model."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "build/myds4")
    args = parser.parse_args()

    def encode(text, flags):
        result = subprocess.run([str(args.binary), "-m", str(args.checkpoint), "-p", text, *flags],
                                capture_output=True, check=True, timeout=30)
        return [int(value) for value in result.stdout.strip().split(b",")]

    plain = "<｜begin▁of▁sentence｜><｜User｜>Review this code.<｜Assistant｜><think>"
    literal = plain + ' const char *image = "<｜deepseek_image｜>";'
    raw = encode(literal, ["--raw", "--tokenize"])
    rendered = encode(literal, ["--rendered", "--tokenize"])
    assert 129264 in raw, "raw input must retain its explicit image-token spelling"
    assert 129264 not in rendered, "rendered source text must not become an image placeholder"
    assert len(rendered) > len(raw), "literal spelling must be tokenized as text"
    assert rendered == encode(literal, ["--tokenize", "--rendered"]), "tokenize flag order changed the input mode"
    assert encode(plain, ["--raw", "--tokenize"]) == encode(plain, ["--rendered", "--tokenize"])
    assert rendered == encode(literal, ["--rendered", "--perplexity", "--tokenize"]), "perplexity must preserve explicit rendered input"
    assert rendered == encode(literal, ["--perplexity", "--rendered", "--tokenize"]), "rendered input must work after perplexity"
    print("rendered prompt: 7/7 CPU checks passed")


if __name__ == "__main__":
    main()
