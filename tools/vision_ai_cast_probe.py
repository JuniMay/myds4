"""Check whether conversion preserves an explicitly requested BF16 round-trip.

This small CPU-only Core AI check separates export behavior from ANE arithmetic.
It writes one tiny source asset and prints the expected and actual values.
"""

import asyncio
import json
import os
from pathlib import Path

os.environ["USE_OS_COREAI"] = "1"

import numpy as np
import torch
from coreai.runtime import AIModel, NDArray, SpecializationOptions
from coreai_torch import TorchConverter, get_decomp_table


class RoundTrip(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.bfloat16).to(torch.float16)


async def main():
    torch.set_num_threads(1)
    x = torch.tensor(
        [1.0009765625, 1.00390625, 1.0048828125, 1.0078125], dtype=torch.float16
    ).reshape(1, 1, 1, 4)
    model = RoundTrip().eval()
    expected = model(x).numpy()
    exported = torch.export.export(model, (x,)).run_decompositions(get_decomp_table())
    program = (
        TorchConverter()
        .add_exported_program(exported, input_names=["x"], output_names=["y"])
        .to_coreai()
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "build"
        / "vision-models"
        / "diagnostics"
        / "bf16-roundtrip.aimodel"
    )
    program.save_asset(path)
    runtime = await AIModel.load(path, SpecializationOptions.cpu_only())
    outputs = await runtime.load_function("main")(inputs={"x": NDArray(x.numpy())})
    actual = outputs["y"].numpy()
    print(
        json.dumps(
            {
                "input": x.flatten().tolist(),
                "expected": expected.flatten().tolist(),
                "coreai_cpu": actual.flatten().tolist(),
                "preserved_rounding": bool(np.array_equal(expected, actual)),
            }
        )
    )
    if not np.array_equal(expected, actual):
        raise SystemExit("Core AI conversion did not preserve the requested BF16 rounding")


if __name__ == "__main__":
    asyncio.run(main())
