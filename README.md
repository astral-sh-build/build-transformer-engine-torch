# build-transformer-engine-torch

Pre-built Linux wheels for the PyTorch extension from
[NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine), across
Python, PyTorch, CUDA, and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index
for each CUDA version. Each wheel has a local version suffix that identifies the
CUDA and PyTorch versions it was built against, such as
`transformer-engine-torch==2.15.0+cu.12.8.torch.2.11`, and requires the matching
PyTorch release.

Pre-built wheels are available on
[Astral's GPU indexes](https://wheels.astral.sh/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add transformer-engine-torch --index astral-cu128=https://wheels.astral.sh/simple/cu128/
```

This configures the index and uses it as the source for
`transformer-engine-torch`:

```toml
[tool.uv.sources]
transformer-engine-torch = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astral.sh/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astral.sh/simple/cu128/ transformer-engine-torch
```

## Supported versions

Wheels are available for the following Transformer Engine versions:

- [`2.16.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.16)
- [`2.15.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.15)
- [`2.14.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.14)
- [`2.13.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.13)
- [`2.12.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.12)
- [`2.11.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.11-r1)
- [`2.10.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.10)
- [`2.9.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.9)
- [`2.8.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.8)
- [`2.7.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.7)
- [`2.6.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.6)
- [`2.5.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.5)
- [`2.4.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.4)
- [`2.3.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.3)
- [`2.2.0`](https://github.com/astral-sh-build/build-transformer-engine-torch/releases/tag/v2.2.1)

The latest release, Transformer Engine 2.16.0, supports the following
combinations:

| PyTorch | Python    | `x86_64` CUDA          | `aarch64` CUDA         |
| ------- | --------- | ---------------------- | ---------------------- |
| 2.4.1   | 3.10–3.12 | 12.1, 12.4             | —                      |
| 2.5.1   | 3.10–3.12 | 12.1, 12.4             | —                      |
| 2.6.0   | 3.10–3.12 | 12.4, 12.6             | 12.6                   |
| 2.7.1   | 3.10–3.13 | 12.6, 12.8             | 12.8                   |
| 2.8.0   | 3.10–3.13 | 12.6, 12.8, 12.9       | 12.9                   |
| 2.9.1   | 3.10–3.13 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.11.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.12.1  | 3.10–3.14 | 12.6, 13.0, 13.2       | 12.6, 13.0, 13.2       |
| 2.13.0  | 3.10–3.15 | 12.6, 13.0, 13.2       | 12.6, 13.0, 13.2       |

## License

build-transformer-engine-torch is licensed under the
[Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
