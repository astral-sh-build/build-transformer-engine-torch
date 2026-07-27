import subprocess
from pathlib import Path

import modal

TEST_DIRECTORY = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(TEST_DIRECTORY))
    .run_commands(
        'python -c "import os, sysconfig; '
        "root = os.path.join(sysconfig.get_path('purelib'), 'nvidia'); "
        "os.symlink(os.path.join(root, 'cuda_runtime'), "
        "os.path.join(root, 'cudart'))\""
    )
    .add_local_file(
        TEST_DIRECTORY / "test_transformer_engine_torch.py",
        remote_path="/gpu-tests/test_transformer_engine_torch.py",
    )
)

app = modal.App("astral-build-transformer-engine-torch-gpu-tests")


@app.function(image=image, gpu="H100", timeout=900)
def test() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-v", "/gpu-tests/test_transformer_engine_torch.py"],
        check=True,
    )
