import importlib
from importlib.metadata import version

import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    device = torch.device("cuda")
    assert torch.cuda.get_device_capability(device)[0] >= 9
    return device


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("transformer-engine-torch") == "2.16.0+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize(
    "module_name",
    ["transformer_engine", "transformer_engine.pytorch"],
)
@pytest.mark.xfail(
    raises=AssertionError,
    reason="Transformer Engine rejects the published wheel's local version",
    strict=True,
)
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.xfail(
    raises=AssertionError,
    reason="Transformer Engine rejects the published wheel's local version",
    strict=True,
)
def test_cuda_linear_forward_backward(device: torch.device, dtype: torch.dtype) -> None:
    import transformer_engine.pytorch as te

    layer = te.Linear(64, 32, device=device, params_dtype=dtype)
    inputs = torch.randn((8, 64), device=device, dtype=dtype, requires_grad=True)
    actual = layer(inputs)
    assert actual.shape == (8, 32)
    assert torch.isfinite(actual).all()
    actual.float().square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.xfail(
    raises=AssertionError,
    reason="Transformer Engine rejects the published wheel's local version",
    strict=True,
)
def test_cuda_layer_norm(device: torch.device) -> None:
    import transformer_engine.pytorch as te

    layer = te.LayerNorm(64, device=device, params_dtype=torch.float32)
    inputs = torch.randn((8, 64), device=device, requires_grad=True)
    actual = layer(inputs)
    expected = torch.nn.functional.layer_norm(
        inputs, (64,), layer.weight, layer.bias, layer.eps
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().mean().backward()
    assert inputs.grad is not None


def test_upstream_dependency_versions(device: torch.device) -> None:
    assert version("transformer-engine") == "2.16.0"
    assert version("transformer-engine-cu12") == "2.16.0"
