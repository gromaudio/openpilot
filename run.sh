#!/usr/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
ln -sfn $(pwd) /data/pythonpath
export PYTHONPATH="$PWD"
./tools/webcam/accept_terms.py
pushd selfdrive
PASSIVE=0 NOSENSOR=1 WEBCAM=1 GET_CPU_USAGE=1 ./manager.py | grep -v "too small"
popd
