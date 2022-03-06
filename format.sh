#! /bin/bash -eux

for s in xla_to_onnx.py test_xla_to_onnx.py; do
    mypy ${s} --ignore-missing-imports
    isort ${s}
    autopep8 ${s} --in-place
    black ${s}
done
