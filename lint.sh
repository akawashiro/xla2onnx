#! /bin/bash -eux

cd $(git rev-parse --show-toplevel)
MYPYPATH=hlo_proto mypy xla2onnx/xla2onnx.py --ignore-missing-imports --strict
MYPYPATH=hlo_proto mypy xla2onnx/utils_for_test.py --ignore-missing-imports
MYPYPATH=hlo_proto mypy tests/test_resnet.py --ignore-missing-imports
MYPYPATH=hlo_proto mypy tests/test_xla2onnx.py --ignore-missing-imports
