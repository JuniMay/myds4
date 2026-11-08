# Evaluation inputs

This directory contains the GSM8K test split, HumanEval problems, their
licenses and source URLs in `sources.json`. They are evaluation inputs,
not performance records.

`benchmarks/evaluate.py` selects bounded subsets. Its HumanEval syntax and
entry-point checks do not execute generated code and are not pass@1.
Benchmark results go to an explicitly chosen output file outside this
directory. Do not commit generated outputs or model responses here.
