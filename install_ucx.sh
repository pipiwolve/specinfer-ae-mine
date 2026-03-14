#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "$(dirname "${BASH_SOURCE[0]}")"

UCX_TARBALL_URL="${UCX_TARBALL_URL:-https://ghfast.top/https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz}"
if [[ ! -f ucx-1.15.0.tar.gz ]]; then
  wget -O ucx-1.15.0.tar.gz "$UCX_TARBALL_URL" || wget -O ucx-1.15.0.tar.gz https://github.com/openucx/ucx/releases/download/v1.15.0/ucx-1.15.0.tar.gz
fi
tar -xf ucx-1.15.0.tar.gz
cd ucx-1.15.0
export CUDA_PATH=/usr/local/cuda
export PREFIX=$PWD/install
./contrib/configure-release-mt --prefix="$PREFIX" --without-go --enable-mt --with-cuda="$CUDA_PATH"
make -j install

export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
