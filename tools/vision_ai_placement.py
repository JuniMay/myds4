"""Inspect Core AI placement, or verify a normal execution's Instruments trace.

The debug compute plan can differ from normal Swift execution. It is a
diagnostic, not an acceptance test. --trace checks that each recorded model
call overlaps ANE prediction and that the target has no GPU execution.
"""

import argparse
import asyncio
import os
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# Select Apple's installed runtime, which supports hardware delegates.
os.environ["USE_OS_COREAI"] = "1"


async def check(path):
    from coreai.authoring import AIModelAsset
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    from coreai_torch.debugging.compute_plan import ComputePlan

    program = AIModelAsset.load(path).program
    options = SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.neural_engine()
    )
    plan = await ComputePlan.from_program(program, options)
    plan.write_summary(program)
    data = plan.to_dict(program)
    devices = data["device_histogram"]
    print("Device histogram:", devices)
    reasons = sorted(
        {str(reason) for entry in data["entries"] for reason in entry.get("validation", [])}
    )
    for reason in reasons:
        print("ANE validation:", reason)
    print(
        "Debug specialization only. Verify normal execution with --trace before deciding device placement."
    )


def trace_check(path):
    def read(query=None):
        command = ["xcrun", "xctrace", "export", "--input", str(path)]
        command += ["--xpath", query] if query else ["--toc"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return ET.fromstring(result.stdout)

    target = read().find("./run/info/target/process")
    if target is None:
        raise SystemExit("Trace must have one launched or attached target process")
    pid = int(target.get("pid"))

    def table(schema):
        root = read(f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]')
        definitions = {e.get("id"): e for e in root.iter() if e.get("id")}

        def resolve(e):
            return definitions[e.get("ref")] if e.get("ref") else e

        def label(e):
            e = resolve(e)
            return e.get("fmt") or e.text or ""

        def interval(row):
            start = int(resolve(row.find("start-time")).text)
            return start, start + int(resolve(row.find("duration")).text)

        def is_target(row):
            process = row.find("process")
            if process is None:
                return False
            process = resolve(process)
            number = process.find("pid")
            return number is not None and int(resolve(number).text) == pid

        return root.findall(".//row"), label, interval, is_target

    rows, label, interval, is_target = table("CoreAIProfile")
    calls = [
        interval(row)
        for row in rows
        if is_target(row)
        and any(label(e) == "Function Execute" for e in row.findall("event-concept"))
    ]
    rows, label, interval, _ = table("ane-hw-intervals")
    ane = [interval(row) for row in rows if "Prediction" in label(row.find("formatted-label"))]
    matched = [call for call in calls if any(a < call[1] and b > call[0] for a, b in ane)]
    rows, _, interval, is_target = table("metal-gpu-intervals")
    gpu = [interval(row) for row in rows if is_target(row)]
    print(
        json.dumps(
            {
                "model_calls": len(calls),
                "calls_with_ane_activity": len(matched),
                "target_gpu_intervals": len(gpu),
            }
        )
    )
    if not calls or len(matched) != len(calls) or gpu:
        raise SystemExit("Normal-execution ANE trace check failed")
    print(
        "All recorded model calls overlap ANE prediction; no GPU execution was recorded for the target."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path, nargs="?", help="inspect the debug specialization")
    parser.add_argument("--trace", type=Path, help="verify a normal Core AI Instruments recording")
    args = parser.parse_args()
    if bool(args.asset) == bool(args.trace):
        parser.error("provide either an asset or --trace")
    if args.trace:
        trace_check(args.trace)
    else:
        asyncio.run(check(args.asset))
