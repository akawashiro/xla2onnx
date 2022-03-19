#! /bin/bash -eux

protoc --python_out=xla2onnx --mypy_out=quiet:hlo_proto -I=hlo_proto_def hlo_proto_def/xla_data.proto
protoc --python_out=xla2onnx --mypy_out=quiet:hlo_proto -I=hlo_proto_def hlo_proto_def/hlo.proto
