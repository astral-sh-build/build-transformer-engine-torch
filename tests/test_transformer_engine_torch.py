import importlib
import json
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    distributions,
    version,
)
from pathlib import Path

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
def test_native_module(device: torch.device, module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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


def test_astral_dependency_versions(device: torch.device) -> None:
    assert version("transformer-engine") == "2.16.0"
    assert version("transformer-engine-cu12") == "2.16.0+cu.12.8"
    with pytest.raises(PackageNotFoundError):
        version("transformer-engine-cu13")


@pytest.mark.parametrize(
    ("package_name", "build_repository"),
    [
        ("transformer-engine", "build-transformer-engine"),
        ("transformer-engine-cu12", "build-transformer-engine-cu12"),
        ("transformer-engine-torch", "build-transformer-engine-torch"),
    ],
)
def test_astral_wheel_provenance(
    device: torch.device, package_name: str, build_repository: str
) -> None:
    package = distribution(package_name)
    sbom = next(
        (
            file
            for file in package.files or ()
            if file.parts[-2:] == ("sboms", "astral.json")
        ),
        None,
    )
    assert sbom is not None, f"{package_name} is not an Astral-built wheel"

    provenance = json.loads(Path(package.locate_file(sbom)).read_text())
    assert provenance["source"] == {
        "repository": "https://github.com/NVIDIA/TransformerEngine",
        "tag": "v2.16",
        "commit": "4220403e831d29e93868f7793693ea83f6b8b05b",
    }
    assert provenance["build"]["repository"] == (
        f"https://github.com/astral-sh-build/{build_repository}"
    )


def test_version_patch_survives_shared_package_ownership(
    device: torch.device,
) -> None:
    common_module = "transformer_engine/common/__init__.py"
    owners = {
        package.metadata["Name"].replace("_", "-")
        for package in distributions()
        if any(str(file) == common_module for file in package.files or ())
    }
    assert owners == {"transformer-engine", "transformer-engine-cu12"}

    source = Path(
        distribution("transformer-engine").locate_file(common_module)
    ).read_text()
    assert "module_base_version = " in source
    assert 'te_core_version.split("+")[0]' in source
