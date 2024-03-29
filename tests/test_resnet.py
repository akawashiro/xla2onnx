# copyright 2018 google llc
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

"""A mock-up showing a ResNet50 network with training on synthetic data.

This file uses the stax neural network definition library and the optimizers
optimization library.
"""

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import pytest
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import (
    AvgPool,
    BatchNorm,
    Conv,
    Dense,
    FanInSum,
    FanOut,
    Flatten,
    GeneralConv,
    Identity,
    LogSoftmax,
    MaxPool,
    Relu,
    SumPool,
)

from xla2onnx import check_output, translate_and_run

# ResNet blocks compose other layers


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(
        Conv(filters1, (1, 1), strides),
        BatchNorm(),
        Relu,
        Conv(filters2, (ks, ks), padding="SAME"),
        BatchNorm(),
        Relu,
        Conv(filters3, (1, 1)),
        BatchNorm(),
    )
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters

    def make_main(input_shape):
        # the number of output channels depends on the number of input channels
        return stax.serial(
            Conv(filters1, (1, 1)),
            BatchNorm(),
            Relu,
            Conv(filters2, (ks, ks), padding="SAME"),
            BatchNorm(),
            Relu,
            Conv(input_shape[3], (1, 1)),
            BatchNorm(),
        )

    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks


# TODO: Commented out some layers because emitting full ResNet50 is too slow.
def ResNet50(num_classes):
    return stax.serial(
        # TODO: HWCN is difficult to support
        # GeneralConv(("HWCN", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
        GeneralConv(("NHWC", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [64, 64, 256], strides=(1, 1)),
        # IdentityBlock(3, [64, 64]),
        # IdentityBlock(3, [64, 64]),
        ConvBlock(3, [128, 128, 512]),
        # IdentityBlock(3, [128, 128]),
        # IdentityBlock(3, [128, 128]),
        # IdentityBlock(3, [128, 128]),
        # ConvBlock(3, [256, 256, 1024]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # IdentityBlock(3, [256, 256]),
        # ConvBlock(3, [512, 512, 2048]),
        # IdentityBlock(3, [512, 512]),
        # IdentityBlock(3, [512, 512]),
        AvgPool((7, 7)),
        Flatten,
        Dense(num_classes),
        LogSoftmax,
    )


@pytest.mark.parametrize("N,H,W,C", [(2, 4, 4, 2), (8, 56, 56, 64)])
def test_sumpool_NHWC(N, H, W, C):
    test_name = "resnet_sumpool_NHWC"
    rng_key = random.PRNGKey(0)

    input_shape = (N, H, W, C)

    init_fun, predict_fun = SumPool((2, 2))
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-6)


def test_avgpool():
    test_name = "resnet_avgpool"
    rng_key = random.PRNGKey(0)

    batch_size = 4
    height = 14
    width = 14
    channel = 2
    input_shape = (batch_size, height, width, channel)

    init_fun, predict_fun = AvgPool((7, 7))
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-6)


def test_conv1():
    test_name = "resnet_conv1"
    rng_key = random.PRNGKey(0)

    batch_size = 4
    height = 14
    width = 14
    channel = 2
    input_shape = (batch_size, height, width, channel)

    init_fun, predict_fun = GeneralConv(
        ("NHWC", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"
    )
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-6)


def test_convblock():
    test_name = "convblock"
    rng_key = random.PRNGKey(0)

    batch_size = 4
    height = 14
    width = 14
    channel = 2
    input_shape = (batch_size, height, width, channel)

    init_fun, predict_fun = ConvBlock(3, [64, 64, 256], strides=(1, 1))
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-5)


def test_maxpool():
    test_name = "maxpool"
    rng_key = random.PRNGKey(0)

    batch_size = 4
    height = 14
    width = 14
    channel = 2
    input_shape = (batch_size, height, width, channel)

    init_fun, predict_fun = MaxPool((3, 3), strides=(2, 2))
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-6)


def test_bn():
    test_name = "bn"
    rng_key = random.PRNGKey(0)

    batch_size = 8
    height = 4
    width = 4
    channel = 2
    input_shape = (batch_size, height, width, channel)

    init_fun, predict_fun = BatchNorm()
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    check_output(output_values, outputs[0], atol=1e-6)


# TODO: This test fails because of too large ONNX.
@pytest.mark.xfail
def test_resnet():
    test_name = "resnet"
    rng_key = random.PRNGKey(0)

    batch_size = 8
    num_classes = 1001
    image_size = 224
    # TODO: HWCN is difficult to support.
    # input_shape = (image_size, image_size, 3, batch_size)
    input_shape = (batch_size, image_size, image_size, 3)

    init_fun, predict_fun = ResNet50(num_classes)
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    print(output_values.shape)
    check_output(output_values, outputs[0], atol=1e-3, rtol=1e-3)


# TODO: This test fails because of too large ONNX.
@pytest.mark.xfail
def test_resnet_grad():
    test_name = "resnet_grad"
    rng_key = random.PRNGKey(0)

    batch_size = 8
    num_classes = 1001
    image_size = 224
    # TODO: HWCN is difficult to support.
    # input_shape = (image_size, image_size, 3, batch_size)
    input_shape = (batch_size, image_size, image_size, 3)
    target_shape = (batch_size, num_classes)

    init_fun, predict_fun = ResNet50(num_classes)
    _, init_params = init_fun(rng_key, input_shape)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")
    targets = rng.rand(*target_shape).astype("float32")

    def loss(params, batch):
        inputs, targets = batch
        preds = predict_fun(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=0))

    fn = grad(loss)
    input_values = [init_params, (images, targets)]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name, exit_after_emit_hlo=True)

    # check_output(output_values, outputs[0], atol=1e-3, rtol=1e-3)


# if __name__ == "__main__":
#     rng_key = random.PRNGKey(0)
#
#     batch_size = 8
#     num_classes = 1001
#     input_shape = (224, 224, 3, batch_size)
#     step_size = 0.1
#     num_steps = 10
#
#     init_fun, predict_fun = ResNet50(num_classes)
#     _, init_params = init_fun(rng_key, input_shape)
#
#     def loss(params, batch):
#         inputs, targets = batch
#         logits = predict_fun(params, inputs)
#         return -jnp.sum(logits * targets)
#
#     def accuracy(params, batch):
#         inputs, targets = batch
#         target_class = jnp.argmax(targets, axis=-1)
#         predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
#         return jnp.mean(predicted_class == target_class)
#
#     def synth_batches():
#         rng = npr.RandomState(0)
#         while True:
#             images = rng.rand(*input_shape).astype("float32")
#             labels = rng.randint(num_classes, size=(batch_size, 1))
#             onehot_labels = labels == jnp.arange(num_classes)
#             yield images, onehot_labels
#
#     opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
#     batches = synth_batches()
#
#     @jit
#     def update(i, opt_state, batch):
#         params = get_params(opt_state)
#         return opt_update(i, grad(loss)(params, batch), opt_state)
#
#     opt_state = opt_init(init_params)
#     for i in range(num_steps):
#         opt_state = update(i, opt_state, next(batches))
#     trained_params = get_params(opt_state)
