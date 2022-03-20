#! /bin/bash -eux
# After run this script, you must modify generated files around `import`.

protoc --python_out=xla2onnx --mypy_out=quiet:hlo_proto -I=hlo_proto_def hlo_proto_def/xla_data.proto
protoc --python_out=xla2onnx --mypy_out=quiet:hlo_proto -I=hlo_proto_def hlo_proto_def/hlo.proto
