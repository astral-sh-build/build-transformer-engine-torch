#!/bin/bash
# Script to prepare the build environment for TransformerEngine PyTorch.
#
# Example usage:
#   ./prepare_for_build.sh v2.8

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <transformer_engine_version>"
    echo "Example: $0 v2.8"
    exit 1
fi

TRANSFORMER_ENGINE_VERSION=$1

# Ensure that the TransformerEngine version is supported.
if [ ! -d "${ROOT}/build_scripts/patches/${TRANSFORMER_ENGINE_VERSION}" ]; then
    echo "Error: patches/${TRANSFORMER_ENGINE_VERSION} directory does not exist"
    exit 1
fi

# Apply patches.
for patch in "${ROOT}/build_scripts/patches/${TRANSFORMER_ENGINE_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done
