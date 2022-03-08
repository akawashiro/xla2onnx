#! /bin/bash -eux

python3 -m venv myenv 
source myenv/bin/activate
pip3 install -r requirements.txt

for s in xla2onnx.py test_xla2onnx.py test_resnet.py utils_for_test.py; do
    if [ ${s} == "xla2onnx.py" ]; then
        MYPYPATH=hlo_proto mypy ${s} --ignore-missing-imports --strict
    else
        MYPYPATH=hlo_proto mypy ${s} --ignore-missing-imports
    fi
    isort ${s}
    autopep8 ${s} --in-place
    black ${s}
done
