import subprocess
import sys

import jax
import numpy as np
import onnx
import onnxruntime as ort
from jax import grad, jit, random, vmap

from xla2onnx import hlo_proto_to_onnx

sys.path.append("hlo_proto")  # nopep8

import hlo_pb2  # nopep8
import xla_data_pb2  # nopep8


def gen_onnx_inputs(onnx_name: str, input_values):
    m = onnx.load(onnx_name)
    input_names = list(map(lambda x: x.name, m.graph.input))
    inputs = {}
    flattened = []
    for v in input_values:
        # TODO: Dirty hack
        if isinstance(v, list):
            for t in v:
                assert isinstance(t, tuple)
                flattened.extend(list(t))
        else:
            flattened.append(v)
    assert len(input_names) == len(flattened), (
        "len(input_names) = "
        + str(len(input_names))
        + ", len(flattened) = "
        + str(len(flattened))
    )
    for n, v in zip(input_names, flattened):
        inputs[n] = np.array(v)
    return inputs


def translate_and_run(fn, input_values, test_name):
    onnx_name = test_name + ".onnx"

    # TODO(akawashiro): Use inline=True to remove call
    fn_jit = jit(fn, inline=True)
    xla = jax.xla_computation(fn_jit)(*input_values)

    with open(test_name + "_as_hlo_text.txt", "w") as f:
        f.write(xla.as_hlo_text())
    with open(test_name + "_as_hlo_dot_graph.dot", "w") as f:
        f.write(xla.as_hlo_dot_graph())
    dot_cmd = [
        "dot",
        "-Tps",
        test_name + "_as_hlo_dot_graph.dot",
        "-o",
        test_name + "_as_hlo_dot_graph.ps",
    ]
    subprocess.run(dot_cmd)

    hlo_proto = xla.as_serialized_hlo_module_proto()
    with open(test_name + ".hlo_proto", "wb") as f:
        f.write(hlo_proto)

    with open(test_name + ".hlo_proto", "rb") as f:
        hlo_proto_data = f.read()
        hlo_proto = hlo_pb2.HloModuleProto()
        hlo_proto.ParseFromString(hlo_proto_data)

    with open(test_name + "_hlo_proto.txt", "w") as f:
        f.write(str(hlo_proto))

    hlo_proto_to_onnx(hlo_proto, onnx_name)

    inputs = gen_onnx_inputs(onnx_name, input_values)
    ort_sess = ort.InferenceSession(onnx_name)

    for k, v in inputs.items():
        assert isinstance(v, np.ndarray), f"type(inputs[{k}]) = {type(v)}"

    outputs = ort_sess.run(None, inputs)
    return outputs


def check_output(out1, out2, rtol=1e-05, atol=1e-08):
    assert out1.size == out2.size

    # TODO: Fix these values. Maybe incorrect.
    suggest_atol = np.max(np.abs(out1 - out2))
    rd = np.abs(out1 - out2) / np.abs(out2)
    suggest_rtol = np.max(rd[~np.isnan(rd)])

    assert np.allclose(
        out1, out2, rtol=rtol, atol=atol
    ), f"suggest_rtol = {suggest_rtol} suggest_atol = {suggest_atol}"
