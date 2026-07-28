import os
import subprocess
from importlib.metadata import distribution
from pathlib import Path

import modal

TEST_DIRECTORY = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(
        uv_project_dir=str(TEST_DIRECTORY),
        env={"UV_DEFAULT_INDEX": "https://pypi.org/simple"},
    )
    .add_local_file(
        TEST_DIRECTORY / "test_transformer_engine_torch.py",
        remote_path="/gpu-tests/test_transformer_engine_torch.py",
    )
)

app = modal.App("astral-build-transformer-engine-torch-gpu-tests")


def cuda_library_root(package_name: str, library_name: str) -> str:
    package = distribution(package_name)
    libraries = {
        Path(package.locate_file(file))
        for file in package.files or ()
        if file.name.startswith(library_name) and "stubs" not in file.parts
    }
    roots = {library.parent.parent for library in libraries}
    if len(roots) != 1:
        raise RuntimeError(
            f"Expected one {library_name} root in {package_name}; found {roots}"
        )
    return str(roots.pop())


@app.function(image=image, gpu="H100", timeout=900)
def test() -> None:
    environment = os.environ | {
        "NVRTC_HOME": cuda_library_root("nvidia-cuda-nvrtc-cu12", "libnvrtc.so"),
        "CURAND_HOME": cuda_library_root("nvidia-curand-cu12", "libcurand.so"),
    }
    subprocess.run(
        ["python", "-m", "pytest", "-v", "/gpu-tests/test_transformer_engine_torch.py"],
        check=True,
        env=environment,
    )
