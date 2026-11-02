"""Small CPU checks of vision export layout and global attention semantics."""

import unittest
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from vision_ai import attention, rope, rotary


class VisionMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_query_tiles_keep_all_keys(self):
        heads, dim, count = 3, 8, 19
        q, k, v = (torch.randn(1, heads * dim, 1, count) for _ in range(3))

        def standard(x):
            return x.reshape(1, heads, dim, count).transpose(-1, -2)

        want = F.scaled_dot_product_attention(standard(q), standard(k), standard(v))
        want = want.transpose(-1, -2).reshape(1, heads * dim, 1, count)
        for chunk in (1, 7, 32):
            with self.subTest(chunk=chunk):
                torch.testing.assert_close(
                    attention(q, k, v, heads, chunk), want, atol=1e-6, rtol=1e-5
                )

    def test_rotary_uses_rows_and_columns(self):
        height, width, heads, dim = 2, 3, 2, 8
        x = torch.randn(1, heads * dim, 1, height * width)
        cosine, sine = rope(height, width, dim)
        got = rotary(x, cosine, sine, heads)
        want = x.clone()
        for row in range(height):
            for col in range(width):
                position = row * width + col
                for h in range(heads):
                    for d in range(dim // 2):
                        axis = row if d < dim // 4 else col
                        frequency = 10000 ** (-2 * (d % (dim // 4)) / (dim // 2))
                        angle = torch.tensor(axis * frequency)
                        a = x[0, h * dim + d, 0, position]
                        b = x[0, h * dim + d + dim // 2, 0, position]
                        want[0, h * dim + d, 0, position] = a * angle.cos() - b * angle.sin()
                        want[0, h * dim + d + dim // 2, 0, position] = (
                            b * angle.cos() + a * angle.sin()
                        )
        torch.testing.assert_close(got, want)


class VisionBuild(unittest.TestCase):
    def test_default_build_rejects_ai_before_loading_model(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "myds4"
        if not binary.exists():
            self.skipTest("build/myds4 is not built")
        result = subprocess.run(
            [
                str(binary),
                "-m",
                str(root / "build" / "missing-vision-test-model"),
                "-p",
                "test",
                "--vision-ai",
                str(root / "build" / "vision-models"),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--vision-ai requires", result.stderr)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
