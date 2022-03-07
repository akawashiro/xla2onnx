import subprocess
import sys
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import onnx
import onnxruntime as ort
import pytest
from jax import grad, jit, random, vmap
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from onnx import AttributeProto, GraphProto, TypeProto, helper

import datasets
from xla2onnx import hlo_proto_to_onnx

sys.path.append("hlo_proto")  # nopep8

import hlo_pb2  # nopep8
import xla_data_pb2  # nopep8

from utils_for_test import check_output, translate_and_run


def test_mnist():
    test_name = "mnist"
    init_random_params, predict = stax.serial(
        Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax
    )

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    rng = random.PRNGKey(0)
    _, init_params = init_random_params(rng, (-1, 28 * 28))

    fn = predict
    input_values = [init_params, train_images[0]]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_add(shape):
    test_name = "add"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.add
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


# TODO: Test axis
@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_sum(shape):
    test_name = "sum"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.sum
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], rtol=1e-4)


# TODO: Test axis
@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_reduce_max(shape):
    test_name = "reduce_max"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.max
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shapes", [((32, 32), (32,)), ((64, 32, 32), (32,))])
def test_add_broadcast(shapes):
    test_name = "add_broadcast"

    input_values = [
        np.random.normal(size=shapes[0]).astype(np.float32),
        np.random.normal(size=shapes[1]).astype(np.float32),
    ]
    fn = jnp.add
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_sub(shape):
    test_name = "sub"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.subtract
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_max(shape):
    test_name = "maximum"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.maximum
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(((32, 32), (32, 32))),
        pytest.param(((1024, 32), (32, 128))),
        pytest.param(((32,), (32,)), marks=pytest.mark.xfail),
        pytest.param(((64, 32), (32,))),
        pytest.param(((32,), (32, 64))),
    ],
)
def test_dot(shapes):
    test_name = "dot"

    input_values = [
        np.random.normal(size=shapes[0]).astype(np.float32),
        np.random.normal(size=shapes[1]).astype(np.float32),
    ]
    fn = jnp.dot
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], rtol=1e-3)


@pytest.mark.parametrize("shape", [(2, 3)])
def test_constant(shape):
    test_name = "constant"

    input_values = []
    constant = np.random.normal(size=shape).astype(np.float32)

    def fn():
        return constant

    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_exp(shape):
    test_name = "exp"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
    ]
    fn = jnp.exp
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_log(shape):
    test_name = "log"

    x = np.random.normal(size=shape).astype(np.float32)
    input_values = [x - np.min(x)]
    fn = jnp.log
    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


@pytest.mark.parametrize("shape", [(32, 32), (32, 64)])
def test_add_exp(shape):
    test_name = "add_exp"

    input_values = [
        np.random.normal(size=shape).astype(np.float32),
        np.random.normal(size=shape).astype(np.float32),
    ]

    def fn(x, y):
        return jnp.exp(jnp.add(x, y))

    output_values = fn(*input_values)

    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0])


# Copied from onnx/backend/test/case/node/__init__.py
def _extract_value_info(
    input: Union[List[Any], np.ndarray, None],
    name: str,
    type_proto: Optional[TypeProto] = None,
) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError(
                "_extract_value_info: both input and type_proto arguments cannot be None."
            )
        elif isinstance(input, list):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype]
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        else:
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype]
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)


# Prepare for general reduce support.


def test_onnx_loop():
    # Given a tensor x of values [x1, ..., xN],
    # Return a sequence of tensors of
    #   [[x1], [x1, x2], ..., [x1, ..., xN]]

    seq_in = onnx.helper.make_tensor_sequence_value_info(
        "seq_in", onnx.TensorProto.FLOAT, None
    )
    seq_out = onnx.helper.make_tensor_sequence_value_info(
        "seq_out", onnx.TensorProto.FLOAT, None
    )
    cond_in = onnx.helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info(
        "iter_count", onnx.TensorProto.INT64, []
    )

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

    x_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["x"],
        value=onnx.helper.make_tensor(
            name="const_tensor_x",
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        ),
    )

    one_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one"],
        value=onnx.helper.make_tensor(
            name="const_tensor_one", data_type=onnx.TensorProto.INT64, dims=(), vals=[1]
        ),
    )

    zero_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["slice_start"],
        value=onnx.helper.make_tensor(
            name="const_tensor_zero",
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=[0],
        ),
    )

    axes_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes"],
        value=onnx.helper.make_tensor(
            name="const_tensor_axes",
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[0],
        ),
    )

    add_node = onnx.helper.make_node(
        "Add", inputs=["iter_count", "one"], outputs=["end"]
    )

    end_unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
    )

    slice_node = onnx.helper.make_node(
        "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
    )

    insert_node = onnx.helper.make_node(
        "SequenceInsert", inputs=["seq_in", "slice_out"], outputs=["seq_out"]
    )

    identity_node = onnx.helper.make_node(
        "Identity", inputs=["cond_in"], outputs=["cond_out"]
    )

    loop_body = onnx.helper.make_graph(
        [
            identity_node,
            x_const_node,
            one_const_node,
            zero_const_node,
            add_node,
            axes_node,
            end_unsqueeze_node,
            slice_node,
            insert_node,
        ],
        "loop_body",
        [iter_count, cond_in, seq_in],
        [cond_out, seq_out],
    )

    node = onnx.helper.make_node(
        "Loop",
        inputs=["trip_count", "cond", "seq_empty"],
        outputs=["seq_res"],
        body=loop_body,
    )

    trip_count = np.array(5).astype(np.int64)
    seq_empty: List[Any] = []
    seq_res = [x[: int(i)] for i in x]
    cond = np.array(1).astype(bool)

    trip_count_info = _extract_value_info(trip_count, "trip_count")
    cond_info = _extract_value_info(trip_count, "cond")
    seq_empty_info = _extract_value_info(trip_count, "seq_empty")
    seq_res_info = _extract_value_info(seq_res, "seq_res")

    graph_def = helper.make_graph(
        [node],
        "test-model",
        [trip_count_info, cond_info, seq_empty_info],
        [seq_res_info],
    )
    model_def = helper.make_model(
        graph_def, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model_def)
    onnx.save(model_def, "onnx_loop.onnx")


# TODO: Cannot test broadcast_to alone. Instead of this, I made add_broadcast
# test.
# @pytest.mark.parametrize("shapes", [([32], [64, 32])])
# def test_broadcast(shapes):
#     print(shapes)
#     test_name = "broadcast"
#
#     x = np.random.normal(size=shapes[0]).astype(np.float32)
#     input_values = [x, shapes[1]]
#     fn = jnp.broadcast_to
#     output_values = fn(*input_values)
#
#     outputs = translate_and_run(fn, input_values, test_name)
#     # assert output_values.shape == shapes[1]
#     # assert np.allclose(output_values, outputs[0])
