# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A mock-up showing a ResNet50 network with training on synthetic data.

This file uses the stax neural network definition library and the optimizers
optimization library.
"""

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
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
)

from utils_for_test import check_output, translate_and_run

# ResNet blocks compose other layers


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(
        Conv(filters1, (1, 1), strides),
        # BatchNorm(),
        # Relu,
        # Conv(filters2, (ks, ks), padding="SAME"),
        # BatchNorm(),
        # Relu,
        Conv(filters3, (1, 1)),
        # BatchNorm(),
    )
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
    Shortcut1 = stax.serial(Conv(filters3, (1, 1), strides))
    Shortcut2 = stax.serial(Conv(filters3, (1, 1), strides))
    # return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)
    return stax.serial(FanOut(2), stax.parallel(Shortcut1, Shortcut2), FanInSum)


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
        # ConvBlock(3, [128, 128, 512]),
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
        # AvgPool((7, 7)),
        # Flatten,
        # Dense(num_classes),
        # LogSoftmax,
    )


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

    check_output(output_values, outputs[0], atol=1e-6)


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


def test_resnet():
    test_name = "resnet"
    rng_key = random.PRNGKey(0)

    batch_size = 8
    num_classes = 1001
    # TODO: HWCN is difficult to support
    # input_shape = (224, 224, 3, batch_size)
    # input_shape = (batch_size, 224, 224, 3)
    input_shape = (batch_size, 6, 6, 3)

    init_fun, predict_fun = ResNet50(num_classes)
    _, init_params = init_fun(rng_key, input_shape)
    # print(len(init_params[0]))
    # init_params = [(jnp.ones(init_params[0][0].shape, dtype=init_params[0][0].dtype), jnp.ones(init_params[0][1].shape, dtype=init_params[0][1].dtype))]
    # for p in init_params:
    #     p.fill(1)
    # print(init_params)

    rng = npr.RandomState(0)
    images = rng.rand(*input_shape).astype("float32")
    # images = np.ones(input_shape, dtype=np.float32)

    fn = predict_fun
    input_values = [init_params, images]

    output_values = fn(*input_values)
    outputs = translate_and_run(fn, input_values, test_name)

    print(output_values.shape)
    # check_output(output_values[0][0], outputs[0][0][0])
    check_output(output_values, outputs[0], atol=1e-5, rtol=1e-3)


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
