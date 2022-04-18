# copyright 2022 Akira Kawata
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     https://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

# TODO(akawashiro): I want to put this file in tests directory

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from jax import grad, jit, random, vmap

from xla2onnx import hlo_proto_to_onnx

sys.path.append("hlo_proto")  # nopep8

from . import hlo_pb2  # nopep8
from . import xla_data_pb2  # nopep8


def flatten_inputs(inputs):
    if isinstance(inputs, np.ndarray):
        return [inputs]
    elif isinstance(inputs, jnp.ndarray):
        return [np.array(inputs)]
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        return sum(map(flatten_inputs, inputs), [])
    else:
        raise RuntimeError("flatten_inputs: " + str(inputs))


def gen_onnx_inputs(onnx_name: str, input_values):
    m = onnx.load(onnx_name)
    input_names = list(map(lambda x: x.name, m.graph.input))
    inputs = {}
    flattened = flatten_inputs(input_values)

    assert len(input_names) == len(flattened), (
        "len(input_names) = "
        + str(len(input_names))
        + ", len(flattened) = "
        + str(len(flattened))
    )
    for n, v in zip(input_names, flattened):
        inputs[n] = np.array(v)
    return inputs


def translate_and_run(fn, input_values, test_name, exit_after_emit_hlo=False):
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
        "-Tsvg",
        test_name + "_as_hlo_dot_graph.dot",
        "-o",
        test_name + "_as_hlo_dot_graph.svg",
    ]
    subprocess.run(dot_cmd)

    if exit_after_emit_hlo:
        return

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


def check_output(out1, out2, rtol=1e-05, atol=1e-08, equal_nan=False):
    assert out1.size == out2.size

    # TODO: Fix these values. Maybe incorrect.
    suggest_atol = np.max(np.abs(out1 - out2))
    rd = np.abs(out1 - out2) / np.abs(out2)
    suggest_rtol = np.max(rd[~np.isnan(rd)])

    error_msg = f"out1.shape = {out1.shape} out2.shape = {out2.shape} suggest_rtol = {suggest_rtol} suggest_atol = {suggest_atol}"
    if out1.size < 64:
        error_msg += f" out1 = {str(out1)} out2 = {str(out2)}"

    assert np.allclose(out1, out2, rtol=rtol, atol=atol, equal_nan=equal_nan), error_msg
