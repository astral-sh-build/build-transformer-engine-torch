#!/bin/bash
# Script to prepare the build environment for Transformer Engine (PyTorch).
#
# Example usage:
#   ./prepare_for_build.sh v2.2.1

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <transformer_engine_version>"
    echo "Example: $0 v2.2.1"
    exit 1
fi

TRANSFORMER_ENGINE_VERSION=$1

# Apply patches.
patch_dir="${ROOT}/build_scripts/patches/${TRANSFORMER_ENGINE_VERSION}"

# Not all Transformer Engine versions need patches.
if [ ! -d "${patch_dir}" ]; then
    echo "Warning: nothing to patch: patches/${TRANSFORMER_ENGINE_VERSION} directory does not exist"
else
    for patch in "${patch_dir}"/*.patch; do
        # Skip if no patch files exist (only .gitkeep)
        if [ -f "${patch}" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi
