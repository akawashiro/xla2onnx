#! /bin/bash -eux

# python3 -m venv myenv 
# source myenv/bin/activate
pip3 install -r requirements.txt

# Generate hlo_proto
rm -rf hlo_proto
mkdir hlo_proto
protoc --python_out=hlo_proto -I=hlo_proto_def hlo_proto_def/xla_data.proto
protoc --python_out=hlo_proto -I=hlo_proto_def hlo_proto_def/hlo.proto

python3 -m pytest -s test_xla2onnx.py
