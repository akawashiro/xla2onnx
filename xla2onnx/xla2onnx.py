import copy
import subprocess
import sys
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import pytest
from jax import grad, jit, vmap
from onnx import AttributeProto, GraphProto, TensorProto, TypeProto, helper

from . import hlo_pb2  # nopep8
from . import xla_data_pb2  # nopep8


# TODO: Remove workaround of type annotation
# https://stackoverflow.com/questions/63893782/python3-raising-attribute-error-on-type-annotation
def translate_dtype(
    element_type: "xla_data_pb2.PrimitiveType.ValueType",
) -> Any:
    assert element_type in [xla_data_pb2.F32]
    if element_type == xla_data_pb2.F32:
        return TensorProto.FLOAT


def shape_proto_to_zeros(
    name: str, shape_proto: xla_data_pb2.ShapeProto
) -> TensorProto:
    dims = shape_proto.dimensions
    dtype = translate_dtype(shape_proto.element_type)
    zeros = np.zeros(dims)
    return helper.make_tensor(name, data_type=dtype, dims=dims, vals=zeros)


def constant_tensor(name: str, constant: npt.NDArray[np.float32]) -> TensorProto:
    return helper.make_tensor(
        name, data_type=TensorProto.FLOAT, dims=constant.shape, vals=constant
    )


def shape_proto_to_value_info_proto(
    name: str, shape_proto: xla_data_pb2.ShapeProto
) -> onnx.ValueInfoProto:
    dims = shape_proto.dimensions
    dtype = translate_dtype(shape_proto.element_type)
    return helper.make_tensor_value_info(name, dtype, dims)


def translate_inputs(parameters: List[Any]) -> Tuple[List[str], List[onnx.TensorProto]]:
    names = []
    values = []
    for i in range(len(parameters)):
        names.append("input" + str(i))
        values.append(shape_proto_to_value_info_proto("input" + str(i), parameters[i]))
    return (names, values)


def translate_outputs(tuple_shapes: List[Any]) -> Tuple[List[str], List[Any]]:
    names = []
    values = []
    for i in range(len(tuple_shapes)):
        names.append("output" + str(i))
        values.append(
            shape_proto_to_value_info_proto("output" + str(i), tuple_shapes[i])
        )
    return (names, values)


gensym_id: int = 0


def gensym(prefix: str = "") -> str:
    global gensym_id
    gensym_id += 1
    return prefix + "gensym_" + str(gensym_id)


def is_sum_reduce_op(reduce_op: hlo_pb2.HloComputationProto) -> bool:
    return (
        len(reduce_op.instructions) == 4 and reduce_op.instructions[3].opcode == "add"
    )


def is_max_reduce_op(reduce_op: hlo_pb2.HloComputationProto) -> bool:
    return (
        len(reduce_op.instructions) == 4
        and reduce_op.instructions[3].opcode == "maximum"
    )


def get_computation(
    hlo_proto: hlo_pb2.HloModuleProto, computation_id: int
) -> hlo_pb2.HloComputationProto:
    for c in hlo_proto.computations:
        if c.id == computation_id:
            return c
    raise RuntimeError("Cannot find computation of " + str(computation_id))


def get_instruction(
    computation: hlo_pb2.HloComputationProto, instruction_id: int
) -> hlo_pb2.HloInstructionProto:
    for i in computation.instructions:
        if i.id == instruction_id:
            return i
    raise RuntimeError("Cannot find instruction of " + str(instruction_id))


def sumpool_HW(
    computation: hlo_pb2.HloComputationProto,
    instruction: hlo_pb2.HloInstructionProto,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    assert len(instruction.operand_ids) == 2
    image_id = str(instruction.operand_ids[0])
    image_shape = get_instruction(computation, instruction.operand_ids[0]).shape

    assert len(instruction.window.dimensions) == 2
    assert len(instruction.shape.dimensions) == 2
    d0 = instruction.window.dimensions[0]
    d1 = instruction.window.dimensions[1]

    ret = []

    reshape_to_NCHW_shape_id = gensym("reshape_to_NCHW_shape_")
    reshape_to_NCHW_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[reshape_to_NCHW_shape_id],
        value=helper.make_tensor(
            gensym("reshape_to_NCHW_shape_tensor_"),
            data_type=TensorProto.INT64,
            dims=[4],
            # TODO: Maybe incorrect
            vals=[
                1,
                1,
                image_shape.dimensions[0],
                image_shape.dimensions[1],
            ],
        ),
    )
    ret.append((reshape_to_NCHW_shape_id, None, reshape_to_NCHW_shape_node))

    # HW -> NCHW
    reshape_to_NCHW_id = gensym("reshape_to_NCHW_")
    reshape_to_NCHW_node = onnx.helper.make_node(
        "Reshape",
        inputs=[image_id, reshape_to_NCHW_shape_id],
        outputs=[reshape_to_NCHW_id],
    )
    ret.append((reshape_to_NCHW_id, None, reshape_to_NCHW_node))

    kernel_shape = [d0.size, d1.size]
    strides = [d0.stride, d1.stride]
    avgpool_id = gensym("sumpool_avgpool_")
    avgpool_node = onnx.helper.make_node(
        "AveragePool",
        inputs=[reshape_to_NCHW_id],
        outputs=[avgpool_id],
        kernel_shape=kernel_shape,
        strides=strides,
    )
    ret.append((avgpool_id, None, avgpool_node))

    reshape_to_HW_shape_id = gensym("reshape_to_HW_shape_")
    reshape_to_HW_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[reshape_to_HW_shape_id],
        value=helper.make_tensor(
            gensym("reshape_to_HW_shape_tensor_"),
            data_type=TensorProto.INT64,
            dims=[2],
            # TODO: Maybe incorrect
            vals=[instruction.shape.dimensions[0], instruction.shape.dimensions[0]],
        ),
    )
    ret.append((reshape_to_HW_shape_id, None, reshape_to_HW_shape_node))

    reshape_to_HW_id = gensym("reshape_to_HW_")
    reshape_to_HW_node = onnx.helper.make_node(
        "Reshape",
        inputs=[avgpool_id, reshape_to_HW_shape_id],
        outputs=[reshape_to_HW_id],
    )
    ret.append((reshape_to_HW_id, None, reshape_to_HW_node))

    avgpool_mul_constant_id = gensym("sumpool_mul_constant_")
    avgpool_mul_constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[avgpool_mul_constant_id],
        value=constant_tensor(
            gensym("sumpool_mul_constant_value_"),
            np.array([d0.size * d1.size], dtype=np.float32),
        ),
    )
    ret.append((avgpool_mul_constant_id, None, avgpool_mul_constant_node))

    avgpool_mul_id = str(instruction.id)
    avgpool_mul_node = helper.make_node(
        "Mul",
        inputs=[reshape_to_HW_id, avgpool_mul_constant_id],
        outputs=[avgpool_mul_id],
    )
    ret.append((avgpool_mul_id, None, avgpool_mul_node))

    return ret


def sumpool_NHWC(
    instruction: hlo_pb2.HloInstructionProto,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    assert len(instruction.operand_ids) == 2
    image_id = str(instruction.operand_ids[0])

    # TODO: Support only classical SumPool
    assert len(instruction.window.dimensions) == 4
    d0 = instruction.window.dimensions[0]
    d1 = instruction.window.dimensions[1]
    d2 = instruction.window.dimensions[2]
    d3 = instruction.window.dimensions[3]
    assert (
        d0.size == 1
        and d0.stride == 1
        and d0.window_dilation == 1
        and d0.base_dilation == 1
    )
    assert (
        d3.size == 1
        and d3.stride == 1
        and d3.window_dilation == 1
        and d3.base_dilation == 1
    )

    # NHWC -> NCHW
    transpose_image_id = gensym("sumpool_transpose_image_")
    transpose_image_node = onnx.helper.make_node(
        "Transpose",
        inputs=[image_id],
        outputs=[transpose_image_id],
        perm=np.array([0, 3, 1, 2]),
    )

    kernel_shape = [d1.size, d2.size]
    strides = [d1.stride, d2.stride]
    avgpool_id = gensym("sumpool_avgpool_")
    avgpool_node = onnx.helper.make_node(
        "AveragePool",
        inputs=[transpose_image_id],
        outputs=[avgpool_id],
        kernel_shape=kernel_shape,
        strides=strides,
    )

    # NCHW -> NHWC
    transpose_output_id = gensym("sumpool_avgpool_output_")
    transpose_output_node = onnx.helper.make_node(
        "Transpose",
        inputs=[avgpool_id],
        outputs=[transpose_output_id],
        perm=np.array([0, 2, 3, 1]),
    )

    avgpool_mul_constant_id = gensym("sumpool_mul_constant_")
    avgpool_mul_constant_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[avgpool_mul_constant_id],
        value=constant_tensor(
            gensym("sumpool_mul_constant_value_"),
            np.array([d1.size * d2.size], dtype=np.float32),
        ),
    )

    avgpool_mul_id = str(instruction.id)
    avgpool_mul_node = helper.make_node(
        "Mul",
        inputs=[transpose_output_id, avgpool_mul_constant_id],
        outputs=[avgpool_mul_id],
    )

    return [
        (transpose_image_id, None, transpose_image_node),
        (avgpool_id, None, avgpool_node),
        (transpose_output_id, None, transpose_output_node),
        (avgpool_mul_constant_id, None, avgpool_mul_constant_node),
        (avgpool_mul_id, None, avgpool_mul_node),
    ]


# Instruction -> [(name, ValueInfo, Node)]
def t_instruction(
    hlo_proto: hlo_pb2.HloModuleProto,
    computation: hlo_pb2.HloComputationProto,
    instruction: hlo_pb2.HloInstructionProto,
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    # XLA: https://www.tensorflow.org/xla/operation_semantics
    # ONNX: https://github.com/onnx/onnx/blob/main/docs/Operators.md
    if instruction.opcode == "parameter":
        name = str(instruction.id)
        value = shape_proto_to_value_info_proto(str(instruction.id), instruction.shape)
        return [(name, value, None)]
    elif instruction.opcode == "constant":
        if instruction.shape.element_type == xla_data_pb2.F32:
            node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[str(instruction.id)],
                value=helper.make_tensor(
                    gensym("dot_op2_reshape_tensor_"),
                    data_type=TensorProto.FLOAT,
                    dims=list(instruction.shape.dimensions),
                    vals=np.array(instruction.literal.f32s),
                ),
            )
            return [(str(instruction.id), None, node)]
        elif instruction.shape.element_type == xla_data_pb2.PRED:
            # TODO: Currently, it works without this kind of constant. I don't
            # know why...
            return []
        else:
            raise RuntimeError(
                "element_type other than F32 is not supported yet: "
                + str(instruction.shape.element_type)
            )
    elif instruction.opcode == "add":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Add", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "multiply":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Mul", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "rsqrt":
        assert len(instruction.operand_ids) == 1
        input_id = str(instruction.operand_ids[0])

        sqrt_id = gensym("rsqrt_sqrt_")
        sqrt_node = helper.make_node(
            "Sqrt",
            inputs=[input_id],
            outputs=[sqrt_id],
        )

        ones_id = gensym("rsqrt_ones_")
        ones_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[ones_id],
            value=helper.make_tensor(
                gensym("rsqrt_ones_tensor_"),
                data_type=TensorProto.FLOAT,
                dims=list(instruction.shape.dimensions),
                vals=np.ones(
                    np.prod(list(instruction.shape.dimensions)), dtype=np.float32
                ),
            ),
        )

        div_id = str(instruction.id)
        div_node = helper.make_node(
            "Div",
            inputs=[ones_id, sqrt_id],
            outputs=[div_id],
        )

        return [
            (sqrt_id, None, sqrt_node),
            (ones_id, None, ones_node),
            (div_id, None, div_node),
        ]
    elif instruction.opcode == "divide":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Div", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "subtract":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Sub", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "maximum":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Max", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "exponential":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Exp", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "log":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Log", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "dot":
        assert len(instruction.operand_ids) == 2
        op1_dim = get_instruction(
            computation, instruction.operand_ids[0]
        ).shape.dimensions
        op2_dim = get_instruction(
            computation, instruction.operand_ids[1]
        ).shape.dimensions
        if len(op1_dim) == 2 and len(op2_dim) == 2:
            assert op1_dim[1] == op2_dim[0], "Must be Matrix-Matrix multiplication"
            inputs = list(map(lambda x: str(x), instruction.operand_ids))
            node = helper.make_node("Gemm", inputs, [str(instruction.id)])
            return [(str(instruction.id), None, node)]
        elif len(op1_dim) == 2 and len(op2_dim) == 1:
            assert op1_dim[1] == op2_dim[0], "Must be Matrix-Vector multiplication"

            op1_name = str(instruction.operand_ids[0])
            op2_name = str(instruction.operand_ids[1])

            op2_shape_id = gensym("dot_op2_reshape_shape_")
            op2_shape_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[op2_shape_id],
                value=helper.make_tensor(
                    gensym("dot_op2_reshape_tensor_"),
                    data_type=TensorProto.INT64,
                    dims=[2],
                    vals=[op2_dim[0], 1],
                ),
            )

            op2_reshape_id = gensym("dot_op2_reshape_")
            op2_reshape_node = helper.make_node(
                "Reshape", inputs=[op2_name, op2_shape_id], outputs=[op2_reshape_id]
            )

            gemm_result_id = gensym("dot_gemm_")
            gemm_node = helper.make_node(
                "Gemm", [op1_name, op2_reshape_id], [gemm_result_id]
            )

            result_shape_id = gensym("dot_result_reshape_shape_")
            result_shape_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[result_shape_id],
                value=helper.make_tensor(
                    gensym("dot_result_reshape_tensor_"),
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[op1_dim[0]],
                ),
            )

            result_reshape_id = str(instruction.id)
            result_reshape_node = helper.make_node(
                "Reshape",
                inputs=[gemm_result_id, result_shape_id],
                outputs=[result_reshape_id],
            )

            return [
                (op2_shape_id, None, op2_shape_node),
                (op2_reshape_id, None, op2_reshape_node),
                (gemm_result_id, None, gemm_node),
                (result_shape_id, None, result_shape_node),
                (result_reshape_id, None, result_reshape_node),
            ]
        elif len(op1_dim) == 1 and len(op2_dim) == 2:
            assert op1_dim[0] == op2_dim[0], "Must be Vector-Matrix multiplication"

            op1_name = str(instruction.operand_ids[0])
            op2_name = str(instruction.operand_ids[1])

            op1_shape_id = gensym("dot_op1_reshape_shape_")
            op1_shape_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[op1_shape_id],
                value=helper.make_tensor(
                    gensym("dot_op1_reshape_tensor_"),
                    data_type=TensorProto.INT64,
                    dims=[2],
                    vals=[1, op1_dim[0]],
                ),
            )

            op1_reshape_id = gensym("dot_op1_reshape_")
            op1_reshape_node = helper.make_node(
                "Reshape", inputs=[op1_name, op1_shape_id], outputs=[op1_reshape_id]
            )

            gemm_result_id = gensym("dot_gemm_")
            gemm_node = helper.make_node(
                "Gemm", [op1_reshape_id, op2_name], [gemm_result_id]
            )

            result_shape_id = gensym("dot_result_reshape_shape_")
            result_shape_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[result_shape_id],
                value=helper.make_tensor(
                    gensym("dot_result_reshape_tensor_"),
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[op2_dim[1]],
                ),
            )

            result_reshape_id = str(instruction.id)
            result_reshape_node = helper.make_node(
                "Reshape",
                inputs=[gemm_result_id, result_shape_id],
                outputs=[result_reshape_id],
            )

            return [
                (op1_shape_id, None, op1_shape_node),
                (op1_reshape_id, None, op1_reshape_node),
                (gemm_result_id, None, gemm_node),
                (result_shape_id, None, result_shape_node),
                (result_reshape_id, None, result_reshape_node),
            ]
        else:
            raise RuntimeError(
                "Unspported pair of dimensions: " + str(op1_dim) + ", " + str(op2_dim)
            )
    elif instruction.opcode == "tuple":
        # TODO:
        assert len(instruction.operand_ids) == 1
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Identity", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "broadcast":
        assert len(instruction.operand_ids) == 1
        input_id = str(instruction.operand_ids[0])
        dest_shape = instruction.shape

        reshape_dims = copy.deepcopy(dest_shape.dimensions)
        for i in range(len(reshape_dims)):
            if i not in list(instruction.dimensions):
                reshape_dims[i] = 1

        ret = []

        shape_id = gensym("broadcast_reshape_shape_")
        shape_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[shape_id],
            value=helper.make_tensor(
                gensym("broadcast_reshape_tensor_"),
                data_type=TensorProto.INT64,
                dims=[len(dest_shape.dimensions)],
                vals=reshape_dims,
            ),
        )
        ret.append((shape_id, None, shape_node))

        reshape_id = gensym("broadcast_reshape_")
        reshape_node = helper.make_node(
            "Reshape", inputs=[input_id, shape_id], outputs=[reshape_id]
        )
        ret.append((reshape_id, None, reshape_node))

        # TODO: Adding dummy broadcasted value is wasteful clearly. I hope
        # post-process remove this dummy value with constant propagation.
        zeros_id = gensym("broadcast_zero_")
        zeros_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[zeros_id],
            value=shape_proto_to_zeros(
                gensym("broadcast_shape_proto_to_zeros_"), instruction.shape
            ),
        )
        print(
            "instruction.id = "
            + str(instruction.id)
            + " zeros_id = "
            + zeros_id
            + " instruction.shape = "
            + str(instruction.shape)
        )
        ret.append((zeros_id, None, zeros_node))

        add_id = str(instruction.id)
        add_node = helper.make_node(
            "Add", [reshape_id, zeros_id], [str(instruction.id)]
        )
        ret.append((add_id, None, add_node))
        return ret
    elif instruction.opcode == "reshape":
        shape_id = gensym("reshape_shape_")
        shape_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[shape_id],
            value=helper.make_tensor(
                gensym("reshape_tensor_"),
                data_type=TensorProto.INT64,
                dims=[len(instruction.shape.dimensions)],
                vals=instruction.shape.dimensions,
            ),
        )
        inputs = list(map(lambda x: str(x), instruction.operand_ids)) + [shape_id]
        node = helper.make_node("Reshape", inputs=inputs, outputs=[str(instruction.id)])
        return [(shape_id, None, shape_node), (str(instruction.id), None, node)]
    elif instruction.opcode == "reduce":
        assert (
            len(instruction.called_computation_ids) == 1
        ), "Calling multiple computations in reduce opcode. It must be strange."
        reduce_op = get_computation(hlo_proto, instruction.called_computation_ids[0])
        if is_sum_reduce_op(reduce_op):
            axes_id = gensym("reduce_sum_axes_")
            axes_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[axes_id],
                value=helper.make_tensor(
                    gensym("reduce_sum_axes_tensor_"),
                    data_type=TensorProto.INT64,
                    dims=[len(instruction.dimensions)],
                    vals=instruction.dimensions,
                ),
            )

            assert len(instruction.operand_ids) == 2
            # TODO: The second oprand of reduce_sum must be 0 as the identity of monoid. We can ignore it for now.
            inputs = list(map(lambda x: str(x), instruction.operand_ids[:1])) + [
                axes_id
            ]
            node = helper.make_node(
                "ReduceSum", inputs, [str(instruction.id)], keepdims=0
            )
            return [(axes_id, None, axes_node), (str(instruction.id), None, node)]
        if is_max_reduce_op(reduce_op):
            assert len(instruction.operand_ids) == 2
            # TODO: The second oprand of reduce_max must be -inf as the
            # identity of monoid. We can ignore it for now.
            inputs = list(map(lambda x: str(x), instruction.operand_ids[:1]))
            axes = list(instruction.dimensions)
            node = helper.make_node(
                "ReduceMax", inputs, [str(instruction.id)], axes=axes, keepdims=0
            )
            return [(str(instruction.id), None, node)]
        raise RuntimeError()
    elif instruction.opcode == "reduce-window":
        assert (
            len(instruction.called_computation_ids) == 1
        ), "Calling multiple computations in reduce opcode. It must be strange."
        reduce_op = get_computation(hlo_proto, instruction.called_computation_ids[0])
        if is_sum_reduce_op(reduce_op):
            if len(instruction.window.dimensions) == 4:
                return sumpool_NHWC(instruction)
            elif len(instruction.window.dimensions) == 2:
                return sumpool_HW(computation, instruction)
            else:
                raise RuntimeError("Not supported reduce window")
        if is_max_reduce_op(reduce_op):
            # TODO: The second oprand of reduce_max must be -inf as the
            # identity of monoid. We can ignore it for now.
            assert len(instruction.operand_ids) == 2
            image_id = str(instruction.operand_ids[0])

            # TODO: Support only classical MaxPool
            assert len(instruction.window.dimensions) == 4
            d0 = instruction.window.dimensions[0]
            d1 = instruction.window.dimensions[1]
            d2 = instruction.window.dimensions[2]
            d3 = instruction.window.dimensions[3]
            assert (
                d0.size == 1
                and d0.stride == 1
                and d0.window_dilation == 1
                and d0.base_dilation == 1
            )
            assert (
                d3.size == 1
                and d3.stride == 1
                and d3.window_dilation == 1
                and d3.base_dilation == 1
            )

            # NHWC -> NCHW
            transpose_image_id = gensym("maxpool_transpose_image_")
            transpose_image_node = onnx.helper.make_node(
                "Transpose",
                inputs=[image_id],
                outputs=[transpose_image_id],
                perm=np.array([0, 3, 1, 2]),
            )

            kernel_shape = [d1.size, d2.size]
            strides = [d1.stride, d2.stride]
            maxpool_id = gensym("maxpool_")
            maxpool_node = onnx.helper.make_node(
                "MaxPool",
                inputs=[transpose_image_id],
                outputs=[maxpool_id],
                kernel_shape=kernel_shape,
                strides=strides,
            )

            # NCHW -> NHWC
            transpose_output_id = str(instruction.id)
            transpose_output_node = onnx.helper.make_node(
                "Transpose",
                inputs=[maxpool_id],
                outputs=[str(instruction.id)],
                perm=np.array([0, 2, 3, 1]),
            )

            return [
                (transpose_image_id, None, transpose_image_node),
                (maxpool_id, None, maxpool_node),
                (transpose_output_id, None, transpose_output_node),
            ]
        raise RuntimeError("This type of reduce-window is not supported yet")
    elif instruction.opcode == "convolution":
        # ONNX convolution takes NHWC image and OIHW weight. So, we must
        # transpose them.
        assert len(instruction.operand_ids) == 2
        cdn = instruction.convolution_dimension_numbers
        image = str(instruction.operand_ids[0])
        weight = str(instruction.operand_ids[1])

        transpose_image_id = gensym("convolution_transpose_image_")
        transpose_image_node = onnx.helper.make_node(
            "Transpose",
            inputs=[image],
            outputs=[transpose_image_id],
            perm=np.array(
                [0, cdn.input_feature_dimension] + list(cdn.input_spatial_dimensions)
            ),
        )

        transpose_weight_id = gensym("convolution_transpose_weight_")
        transpose_weight_node = onnx.helper.make_node(
            "Transpose",
            inputs=[weight],
            outputs=[transpose_weight_id],
            perm=np.array(
                [
                    cdn.kernel_output_feature_dimension,
                    cdn.kernel_input_feature_dimension,
                ]
                + list(cdn.kernel_spatial_dimensions)
            ),
        )

        assert len(instruction.window.dimensions) == 2
        kernel_shape = [
            instruction.window.dimensions[0].size,
            instruction.window.dimensions[1].size,
        ]
        strides = [
            instruction.window.dimensions[0].stride,
            instruction.window.dimensions[1].stride,
        ]
        # TODO: Maybe incorrect
        # assert instruction.window.dimensions[0].padding_high == instruction.window.dimensions[0].padding_low
        pads = [
            instruction.window.dimensions[0].padding_low,
            instruction.window.dimensions[1].padding_low,
            instruction.window.dimensions[0].padding_high,
            instruction.window.dimensions[1].padding_high,
        ]

        convolution_output_id = gensym("convolution_output_")
        convolution_node = onnx.helper.make_node(
            "Conv",
            inputs=[transpose_image_id, transpose_weight_id],
            outputs=[convolution_output_id],
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        )

        # The output is NCHW. So we must transpose it again.
        out_perm = [0, cdn.input_feature_dimension] + list(cdn.input_spatial_dimensions)
        assert sorted(out_perm) == list(range(len(out_perm)))
        out_perm_inv = list(range(len(out_perm)))
        for i in range(len(out_perm)):
            out_perm_inv[out_perm[i]] = i

        transpose_output_id = gensym("convolution_transpose_output_")
        transpose_output_node = onnx.helper.make_node(
            "Transpose",
            inputs=[convolution_output_id],
            outputs=[str(instruction.id)],
            perm=np.array(out_perm_inv),
        )

        return [
            (transpose_image_id, None, transpose_image_node),
            (transpose_weight_id, None, transpose_weight_node),
            (convolution_output_id, None, convolution_node),
            (transpose_output_id, None, transpose_output_node),
        ]
    else:
        raise RuntimeError(instruction.opcode + " is not supported yet!")


# See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#creating-an-onnx-model-using-helper-functions
# Pass hlo_proto also because some operators such as reduce call other sub-computation.
def t_computation(
    hlo_proto: hlo_pb2.HloModuleProto,
    computation: hlo_pb2.HloComputationProto,
    onnx_filename: str,
) -> None:
    name_value_nodes = []
    for i in computation.instructions:
        name_value_nodes.extend(t_instruction(hlo_proto, computation, i))
    input_values = []
    nodes = []
    for n, v, node in name_value_nodes:
        if v is not None:
            input_values.append(v)
        if node is not None:
            nodes.append(node)
    output_values = [
        shape_proto_to_value_info_proto(
            str(computation.root_id), computation.program_shape.result.tuple_shapes[0]
        )
    ]

    graph_def = helper.make_graph(nodes, "test-model", input_values, output_values)

    op = onnx.OperatorSetIdProto()
    op.version = 13
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op]
    )
    onnx.checker.check_model(model_def)
    inferred_model = onnx.shape_inference.infer_shapes(model_def)
    # print(inferred_model)
    # onnx.save(model_def, onnx_filename)
    onnx.save(inferred_model, onnx_filename)


def hlo_proto_to_onnx(hlo_proto: hlo_pb2.HloModuleProto, onnx_filename: str) -> None:
    main_computation = hlo_proto.computations[-1]
    assert (
        hlo_proto.entry_computation_name == main_computation.name
    ), "TODO: Translate only the main computation"
    t_computation(hlo_proto, main_computation, onnx_filename)
