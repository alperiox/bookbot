"""
Step-by-step backpropagation debugger for PyTorch models.

This tool identifies which layer(s) cause "backward through graph twice" errors
by testing backpropagation through each layer individually with multiple passes.

Usage:
    python debug_backward.py --model gpt|gpdt|mlp|hmlp
    python debug_backward.py --model gpdt --num-passes 5 --output report.json
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from net import GPDT, GPT, MLP, HierarchicalMLP


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


@dataclass
class BackwardTestResult:
    """Result of backward pass test for a layer."""

    layer_name: str
    success: bool
    error_message: Optional[str] = None
    failed_on_pass: Optional[int] = None
    output_shape: Optional[Tuple] = None
    requires_grad: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.layer_name,
            "success": self.success,
            "error": self.error_message,
            "failed_on_pass": self.failed_on_pass,
            "shape": list(self.output_shape) if self.output_shape else None,
            "requires_grad": self.requires_grad,
        }


class IntermediateCapture:
    """
    Context manager for capturing intermediate layer outputs via forward hooks.

    Usage:
        with IntermediateCapture(model) as capture:
            output = model(x)
        outputs = capture.get_outputs()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_handles: List = []
        self.captured_outputs: Dict[str, torch.Tensor] = {}
        self.layer_order: List[str] = []

    def _make_hook(self, layer_name: str):
        """Create a hook function for a specific layer."""

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store output info without affecting computation graph
                self.captured_outputs[layer_name] = {
                    "tensor": output,
                    "shape": tuple(output.shape),
                    "requires_grad": output.requires_grad,
                }
                if layer_name not in self.layer_order:
                    self.layer_order.append(layer_name)

        return hook

    def __enter__(self):
        """Register hooks on all modules."""
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                hook = self._make_hook(name)
                handle = module.register_forward_hook(hook)
                self.hook_handles.append(handle)
        return self

    def __exit__(self, *args):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_outputs(self) -> Dict[str, Dict]:
        """Get captured outputs in forward-pass order."""
        return {
            name: self.captured_outputs[name]
            for name in self.layer_order
            if name in self.captured_outputs
        }


class LayerBackwardTester:
    """
    Tests backward pass through individual layers to identify
    where computation graph issues occur.
    """

    # Default model parameters (from main.py)
    DEFAULT_PARAMS = {
        "n_embed": 15,
        "n_hidden": 400,
        "block_size": 16,
        "num_heads": 3,
        "num_blocks": 2,
        "n_layers": 4,
        "n_consecutive": 2,
        "vocab_size": 100,  # Dummy value for testing
    }

    def __init__(self, device: str = "cpu", num_passes: int = 3):
        self.device = device
        self.num_passes = num_passes
        self.results: List[BackwardTestResult] = []
        self.layer_outputs: Dict[str, Dict] = {}

    def create_model(self, model_name: str) -> nn.Module:
        """Instantiate a model by name with default parameters."""
        p = self.DEFAULT_PARAMS

        if model_name == "gpt":
            model = GPT(
                n_embd=p["n_embed"],
                vocab_size=p["vocab_size"],
                num_heads=p["num_heads"],
                num_blocks=p["num_blocks"],
                block_size=p["block_size"],
            )
        elif model_name == "gpdt":
            model = GPDT(
                n_embd=p["n_embed"],
                vocab_size=p["vocab_size"],
                num_heads=p["num_heads"],
                num_blocks=p["num_blocks"],
                block_size=p["block_size"],
            )
        elif model_name == "mlp":
            model = MLP(
                vocab_size=p["vocab_size"],
                block_size=p["block_size"],
                n_embed=p["n_embed"],
                n_hidden=p["n_hidden"],
                n_layers=p["n_layers"],
            )
        elif model_name == "hmlp":
            model = HierarchicalMLP(
                vocab_size=p["vocab_size"],
                n_consecutive=p["n_consecutive"],
                n_embed=p["n_embed"],
                n_hidden=p["n_hidden"],
                block_size=p["block_size"],
                n_layers=p["n_layers"],
            )
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: gpt, gpdt, mlp, hmlp"
            )

        return model.to(self.device)

    def test_full_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Test backward through multiple forward passes on the full model.

        Returns:
            (success, failed_on_pass, error_message)
        """
        model.train()

        for i in range(self.num_passes):
            try:
                _, loss = model(x, y)
                loss.backward()
                model.zero_grad(set_to_none=True)
            except RuntimeError as e:
                return False, i + 1, str(e)

        return True, None, None

    def capture_layer_info(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, Dict]:
        """Run forward pass and capture all layer outputs."""
        model.train()

        with IntermediateCapture(model) as capture:
            _, loss = model(x, y)

        return capture.get_outputs()

    def test_model(
        self,
        model_name: str,
        batch_size: int = 4,
        seq_len: int = 16,
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic on a model.

        Returns dictionary with full report.
        """
        # Create model
        model = self.create_model(model_name)

        # Create sample input
        x = torch.randint(0, self.DEFAULT_PARAMS["vocab_size"], (batch_size, seq_len))
        x = x.to(self.device)

        # Create sample target
        if model_name in ("gpt", "gpdt"):
            y = torch.randint(
                0, self.DEFAULT_PARAMS["vocab_size"], (batch_size, seq_len)
            )
        else:
            y = torch.randint(0, self.DEFAULT_PARAMS["vocab_size"], (batch_size,))
        y = y.to(self.device)

        # First, capture layer info from a single forward pass
        self.layer_outputs = self.capture_layer_info(model, x, y)

        # Test full model backward with multiple passes
        success, failed_on_pass, error_message = self.test_full_model(model, x, y)

        # Build results
        self.results = []
        for layer_name, info in self.layer_outputs.items():
            result = BackwardTestResult(
                layer_name=layer_name,
                success=success,  # If model passed, all layers passed
                error_message=error_message if not success else None,
                failed_on_pass=failed_on_pass if not success else None,
                output_shape=info["shape"],
                requires_grad=info["requires_grad"],
            )
            self.results.append(result)

        # Find first layer that might be problematic
        # (layers with requires_grad that appeared before failure)
        first_failure = None
        if not success:
            # The error likely originates from a layer that stores state
            # We report all layers for investigation
            first_failure = "Model failed on backward pass"

        return {
            "model": model_name,
            "device": self.device,
            "num_passes": self.num_passes,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "overall_success": success,
            "failed_on_pass": failed_on_pass,
            "error_message": error_message,
            "layers": [r.to_dict() for r in self.results],
            "first_failure": first_failure,
            "timestamp": datetime.now().isoformat(),
        }

    def print_report(self, report: Dict[str, Any]):
        """Print formatted terminal report."""
        c = Colors

        print(f"\n{c.BOLD}{'='*80}{c.RESET}")
        print(f"{c.BOLD}BACKWARD PASS DIAGNOSTIC REPORT{c.RESET}")
        print(f"{c.BOLD}{'='*80}{c.RESET}")

        print(f"\n{c.BLUE}Model:{c.RESET} {report['model']}")
        print(f"{c.BLUE}Device:{c.RESET} {report['device']}")
        print(f"{c.BLUE}Num passes:{c.RESET} {report['num_passes']}")
        print(f"{c.BLUE}Batch size:{c.RESET} {report['batch_size']}")
        print(f"{c.BLUE}Seq length:{c.RESET} {report['seq_len']}")

        print(f"\n{c.BOLD}Layer Analysis:{c.RESET}")
        print("-" * 80)

        for layer in report["layers"]:
            if report["overall_success"]:
                status = f"{c.GREEN}[PASS]{c.RESET}"
            else:
                status = f"{c.YELLOW}[????]{c.RESET}"  # Unknown - model failed

            print(f"\n{status} {c.BOLD}{layer['name']}{c.RESET}")
            print(f"  Shape: {layer['shape']}")
            print(f"  Requires grad: {layer['requires_grad']}")

        print(f"\n{c.BOLD}{'='*80}{c.RESET}")

        if report["overall_success"]:
            print(f"{c.GREEN}{c.BOLD}RESULT: ALL BACKWARD PASSES SUCCEEDED{c.RESET}")
            print(
                f"Model completed {report['num_passes']} forward/backward passes without errors."
            )
        else:
            print(f"{c.RED}{c.BOLD}RESULT: BACKWARD PASS FAILED{c.RESET}")
            print(f"{c.RED}Failed on pass: {report['failed_on_pass']}{c.RESET}")
            print(f"\n{c.YELLOW}Error message:{c.RESET}")
            # Truncate long error messages
            error = report["error_message"]
            if len(error) > 500:
                error = error[:500] + "..."
            print(f"  {error}")

            print(f"\n{c.YELLOW}Diagnosis:{c.RESET}")
            print(
                "  The model likely has a layer that stores state across forward passes."
            )
            print("  Look for patterns like:")
            print("    - self.x = x (storing input without detach)")
            print("    - self.some_tensor = computation(...) (storing computed values)")
            print("    - Tensors created in __init__ that are used in forward()")

        print(f"{c.BOLD}{'='*80}{c.RESET}\n")

    def save_json_report(self, report: Dict[str, Any], output_path: str):
        """Save report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step backpropagation debugger for PyTorch models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt", "gpdt", "mlp", "hmlp"],
        help="Model to test: gpt, gpdt, mlp, or hmlp",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence length for testing (default: 16)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=3,
        help="Number of backward passes to test (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backward_debug_report.json",
        help="Output path for JSON report (default: backward_debug_report.json)",
    )

    args = parser.parse_args()

    # Create tester
    tester = LayerBackwardTester(
        device=args.device,
        num_passes=args.num_passes,
    )

    # Run diagnostic
    print(f"\nRunning backward pass diagnostic for '{args.model}' model...")
    print(f"Testing with {args.num_passes} consecutive forward/backward passes...")

    try:
        report = tester.test_model(
            model_name=args.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
    except Exception as e:
        print(f"\n{Colors.RED}Error during testing: {e}{Colors.RESET}")
        sys.exit(1)

    # Output results
    tester.print_report(report)
    tester.save_json_report(report, args.output)

    # Exit code based on result
    sys.exit(0 if report["overall_success"] else 1)


if __name__ == "__main__":
    main()
