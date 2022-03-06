#! /bin/bash -eux

python3 -m venv myenv 
source myenv/bin/activate
pip3 install -r requirements.txt

for s in xla2onnx.py test_xla2onnx.py test_resnet.py utils_for_test.py; do
    mypy ${s} --ignore-missing-imports
    isort ${s}
    autopep8 ${s} --in-place
    black ${s}
done
