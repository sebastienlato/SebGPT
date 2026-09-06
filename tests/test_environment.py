"""Minimal checks for the Phase 0 SebGPT development environment."""

import platform
import sys
import unittest

import torch


class EnvironmentTests(unittest.TestCase):
    def test_expected_runtime_versions(self) -> None:
        self.assertEqual(sys.version_info[:3], (3, 14, 4))
        self.assertEqual(platform.machine(), "arm64")
        self.assertEqual(torch.__version__, "2.14.0")

    def test_cpu_tensor_and_autograd(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = (values**2).sum()

        loss.backward()

        self.assertEqual(loss.item(), 14.0)
        self.assertEqual(values.grad.tolist(), [2.0, 4.0, 6.0])

    def test_mps_tensor_operation_when_available(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable in this execution context")

        values = torch.tensor([1.0, 2.0, 3.0], device="mps")
        doubled = values * 2
        torch.mps.synchronize()

        self.assertEqual(doubled.cpu().tolist(), [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
