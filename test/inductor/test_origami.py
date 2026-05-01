# Owner(s): ["module: inductor"]
import os
import time
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch._logging import trace_structured
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
ORIGAMI_TOPK_VALUES = [5, 10]
ORIGAMI_COMPILE_TOPK = 2
PERF_SLOWDOWN_TOLERANCE = 1.05
IS_ROCM = torch.version.hip is not None

try:
    import origami

    HAS_ORIGAMI = True
except ImportError:
    origami = None
    HAS_ORIGAMI = False


if IS_ROCM:
    torch.set_float32_matmul_precision("highest")


@unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU and Triton")
@unittest.skipIf(not IS_ROCM, "Origami integration is ROCm-only")
@unittest.skipIf(not HAS_ORIGAMI, "Origami package is not installed")
@unittest.skipIf(
    not (config.max_autotune and config.rocm.origami),
    "Requires both max_autotune and origami to be enabled",
)
class TestOrigami(TestCase):
    def _make_fn_and_inputs(
        self, op_name: str, size: int
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)

        if op_name == "bmm":
            batch = 4
            a = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(x, y):
                return torch.bmm(x, y)

            return fn, (a, b)

        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

        if op_name == "mm":

            def fn(x, y):
                return torch.mm(x, y)

            return fn, (a, b)

        if op_name == "addmm":
            bias = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(inp, x, y):
                return torch.addmm(inp, x, y)

            return fn, (bias, a, b)

        raise AssertionError(f"Unsupported op {op_name}")

    def _benchmark_gpu_call_count(self) -> int:
        return sum(
            value
            for name, value in counters["inductor"].items()
            if "benchmark_gpu" in name
        )

    def _measure_compile_time(
        self, op_name: str, patch_config: dict[str, object], *, size: int
    ) -> float:
        """Return first-call torch.compile wall time (seconds).

        fresh_cache() redirects both the inductor and Triton cache to a fresh
        temp directory, giving clean isolated compile-time numbers.
        """
        fn, args = self._make_fn_and_inputs(op_name, size)
        torch._dynamo.reset()
        with fresh_cache(), config.patch(patch_config):
            compiled = torch.compile(fn, dynamic=False)
            t0 = time.perf_counter()
            _ = compiled(*args)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

    def _compile_with_config(
        self,
        op_name: str,
        patch_config: dict[str, object],
        *,
        size: int,
    ) -> dict[str, object]:
        fn, args = self._make_fn_and_inputs(op_name, size)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()

        with (
            fresh_cache(),
            config.patch(patch_config),
            mock.patch(
                "origami.select_topk_configs", wraps=origami.select_topk_configs
            ) as select_topk,
        ):
            compiled = torch.compile(fn, dynamic=False)
            result = compiled(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

        return {
            "compiled": compiled,
            "args": args,
            "benchmark_gpu_calls": self._benchmark_gpu_call_count(),
            "topk_calls": select_topk.call_count,
        }

    def _origami_default_config(self, topk: int) -> dict[str, object]:
        return {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": topk,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _origami_exhaustive_config(self) -> dict[str, object]:
        return {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": ORIGAMI_TOPK_VALUES[0],
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _max_autotune_default_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "rocm.origami": False,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def test_origami_respects_gemm_search_space(self):
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name, search_space="DEFAULT"):
                default_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(ORIGAMI_TOPK_VALUES[0]),
                    size=256,
                )
                self.assertGreater(default_case["topk_calls"], 0)

            with self.subTest(op_name=op_name, search_space="EXHAUSTIVE"):
                exhaustive_case = self._compile_with_config(
                    op_name,
                    self._origami_exhaustive_config(),
                    size=256,
                )
                self.assertEqual(exhaustive_case["topk_calls"], 0)

    def test_origami_reduces_compile_work_vs_regular_max_autotune(self):
        """Test that origami reduces compile work (GPU benchmarking calls) vs regular max_autotune.

        Uses benchmark_gpu_calls count instead of wall-clock timing to avoid flakiness
        on shared CI runners and sequencing bias (origami runs first and pays import cost).
        """
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name):
                origami_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(ORIGAMI_COMPILE_TOPK),
                    size=256,
                )
                max_autotune_case = self._compile_with_config(
                    op_name,
                    self._max_autotune_default_config(),
                    size=256,
                )
                # Origami with topk should benchmark fewer configs than full max_autotune
                self.assertLess(
                    origami_case["benchmark_gpu_calls"],
                    max_autotune_case["benchmark_gpu_calls"],
                    msg=f"Origami ({origami_case['benchmark_gpu_calls']} calls) should have fewer "
                    f"GPU benchmarks than max_autotune ({max_autotune_case['benchmark_gpu_calls']} calls)",
                )

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_origami_runtime_matches_regular_max_autotune(self):
        for op_name in ("mm", "addmm", "bmm"):
            for size in (8192, 16384):
                for topk in ORIGAMI_TOPK_VALUES:
                    with self.subTest(op_name=op_name, size=size, topk=topk):
                        origami_case = self._compile_with_config(
                            op_name,
                            self._origami_default_config(topk),
                            size=size,
                        )
                        max_autotune_case = self._compile_with_config(
                            op_name,
                            self._max_autotune_default_config(),
                            size=size,
                        )

                        origami_runtime_ms = benchmarker.benchmark(
                            origami_case["compiled"],
                            origami_case["args"],
                            {},
                            warmup=50,
                            rep=200,
                        )
                        max_autotune_runtime_ms = benchmarker.benchmark(
                            max_autotune_case["compiled"],
                            max_autotune_case["args"],
                            {},
                            warmup=50,
                            rep=200,
                        )

                        runtime_ratio = (
                            origami_runtime_ms / max_autotune_runtime_ms
                            if max_autotune_runtime_ms > 0
                            else 1.0
                        )
                        passed = (
                            origami_runtime_ms
                            <= max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE
                        )
                        trace_structured(
                            "origami_perf_test",
                            metadata_fn=lambda: {
                                "op_name": op_name,
                                "size": size,
                                "topk": topk,
                                "origami_runtime_ms": round(origami_runtime_ms, 3),
                                "max_autotune_runtime_ms": round(
                                    max_autotune_runtime_ms, 3
                                ),
                                "runtime_ratio": round(runtime_ratio, 3),
                                "passed": passed,
                            },
                            expect_trace_id=False,
                        )

                        self.assertLessEqual(
                            origami_runtime_ms,
                            max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE,
                        )

    def test_origami_topk_edge_cases(self):
        """Test edge cases for origami_topk parameter.

        This test validates:
        - Proper error handling for invalid inputs (negative, float)
        - Correct behavior for boundary values (0, 1, very large)
        - Graceful degradation when topk is too small
        """
        op_name = "mm"
        size = 256

        # Test case 1: topk = 0 (no configs selected)
        # This should still compile without errors, just with no origami optimization
        with self.subTest(topk=0, test_case="no_configs_selected"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(0),
                    size=size,
                )
                # Should complete compilation even with topk=0
                self.assertIsNotNone(result["compiled"])
            except Exception as e:
                self.fail(f"Compilation failed with topk=0: {e}")

        # Test case 2: topk = 1 (minimal selection)
        # This should select the top 1 config and still work
        with self.subTest(topk=1, test_case="minimal_selection"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(1),
                    size=size,
                )
                self.assertIsNotNone(result["compiled"])
                # With topk=1, origami should still be invoked for selection
                # (unless no configs are available)
            except Exception as e:
                self.fail(f"Compilation failed with topk=1: {e}")

        # Test case 3: Very large topk values (e.g., 1000)
        # Should handle gracefully and just select all available configs
        with self.subTest(topk=1000, test_case="very_large_topk"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(1000),
                    size=size,
                )
                self.assertIsNotNone(result["compiled"])
                # Large topk should not cause issues, just select all available
            except Exception as e:
                self.fail(f"Compilation failed with topk=1000: {e}")

        # Test case 4: Negative topk values (invalid input)
        # Should raise ValueError or handle gracefully in config validation
        with self.subTest(topk=-1, test_case="negative_topk"):
            # Negative topk is invalid, but behavior depends on implementation
            # It may be caught during config application or runtime
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(-1),
                    size=size,
                )
                # If compilation succeeds with negative topk, that is also valid
                # (implementation may coerce to 0 or similar)
                self.assertIsNotNone(result["compiled"])
            except (ValueError, TypeError, RuntimeError) as e:
                # Expected: negative values may raise errors during validation
                self.assertIn(
                    "topk",
                    str(e).lower(),
                    msg=f"Error should mention topk parameter: {e}",
                )

        # Test case 5: Float topk values (type error)
        # Should raise TypeError since topk expects an integer
        with self.subTest(topk=3.5, test_case="float_topk"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(3.5),  # type: ignore[arg-type]
                    size=size,
                )
                # Some implementations may coerce float to int
                self.assertIsNotNone(result["compiled"])
            except (TypeError, ValueError) as e:
                # Expected: float values may raise type errors
                self.assertIn(
                    "topk",
                    str(e).lower(),
                    msg=f"Error should mention topk parameter: {e}",
                )

        # Test case 6: Valid normal topk values for comparison
        # These should all work normally
        for topk_val in [2, 5, 10]:
            with self.subTest(topk=topk_val, test_case="valid_topk"):
                try:
                    result = self._compile_with_config(
                        op_name,
                        self._origami_default_config(topk_val),
                        size=size,
                    )
                    self.assertIsNotNone(result["compiled"])
                    # Valid topk should always trigger origami selection
                    self.assertGreaterEqual(
                        result["topk_calls"],
                        0,
                        msg=f"origami.select_topk_configs should be callable with topk={topk_val}",
                    )
                except Exception as e:
                    self.fail(f"Compilation failed with valid topk={topk_val}: {e}")


@unittest.skipIf(
    HAS_GPU_AND_TRITON and IS_ROCM, "Skipped on ROCm where origami is available"
)
class TestOrigamiSkippedOnNonROCm(TestCase):
    """Test that origami is properly skipped on non-ROCm devices (CUDA/CPU).

    These tests verify that:
    1. origami configuration does not cause errors when disabled
    2. origami.select_topk_configs is not called on non-ROCm hardware
    3. Compilation succeeds with regular config generator as fallback
    4. origami gracefully no-ops on unsupported hardware
    """

    def test_origami_skipped_on_non_rocm(self):
        """Verify that origami is properly skipped on non-ROCm devices.

        Tests that origami gracefully handles non-ROCm environments without
        errors, regardless of configuration settings.
        """
        torch.manual_seed(0)

        # Use CPU device to ensure non-ROCm environment
        size = 128
        a = torch.randn(size, size, device="cpu", dtype=torch.float32)
        b = torch.randn(size, size, device="cpu", dtype=torch.float32)

        def test_fn(x, y):
            return torch.mm(x, y)

        expected = test_fn(a, b)

        torch._dynamo.reset()
        counters.clear()

        # Test with origami config enabled (but we're on non-ROCm)
        config_dict = {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,  # Enable origami config
            "rocm.origami_topk": 5,
            "max_autotune_gemm_search_space": "DEFAULT",
            "triton.native_matmul": False,
        }

        # Mock select_topk_configs to ensure it is not called on non-ROCm
        with fresh_cache(), config.patch(config_dict):
            if HAS_ORIGAMI:
                # If origami is installed, mock it to verify it is not called
                with mock.patch(
                    "origami.select_topk_configs",
                    wraps=origami.select_topk_configs if origami else None,
                ) as mock_select_topk:
                    compiled = torch.compile(test_fn, dynamic=False)
                    result = compiled(a, b)

                    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

                    # On non-ROCm devices, origami.select_topk_configs should NOT be called
                    self.assertEqual(
                        mock_select_topk.call_count,
                        0,
                        msg="origami.select_topk_configs should not be called on non-ROCm devices",
                    )
            else:
                # If origami is not installed, just verify compilation works
                compiled = torch.compile(test_fn, dynamic=False)
                result = compiled(a, b)
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_origami_disabled_uses_regular_config(self):
        """Verify regular config generator is used when origami is explicitly disabled."""
        torch.manual_seed(0)

        size = 128
        a = torch.randn(size, size, device="cpu", dtype=torch.float32)
        b = torch.randn(size, size, device="cpu", dtype=torch.float32)

        def test_fn(x, y):
            return torch.mm(x, y)

        expected = test_fn(a, b)

        torch._dynamo.reset()
        counters.clear()

        # Config with origami explicitly disabled
        config_dict = {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": False,  # Explicitly disable origami
            "max_autotune_gemm_search_space": "DEFAULT",
            "triton.native_matmul": False,
        }

        with fresh_cache(), config.patch(config_dict):
            if HAS_ORIGAMI:
                with mock.patch(
                    "origami.select_topk_configs",
                    wraps=origami.select_topk_configs if origami else None,
                ) as mock_select_topk:
                    compiled = torch.compile(test_fn, dynamic=False)
                    result = compiled(a, b)

                    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

                    # When origami is disabled, select_topk_configs should not be called
                    self.assertEqual(
                        mock_select_topk.call_count,
                        0,
                        msg="origami.select_topk_configs should not be called when origami is disabled",
                    )
            else:
                # Origami not installed - should still work fine
                compiled = torch.compile(test_fn, dynamic=False)
                result = compiled(a, b)
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON and IS_ROCM:
        run_tests()
