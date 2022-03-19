#! /bin/bash -eux

cd $(git rev-parse --show-toplevel)

for s in xla2onnx/xla2onnx.py xla2onnx/utils_for_test.py tests/test_resnet.py tests/test_xla2onnx.py; do
    isort ${s}
    autopep8 ${s} --in-place
    black ${s}
done
