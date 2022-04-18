xla2onnx
========
Convert XLA to ONNX.

- Install: `pip3 install -e .`
- Format: `./format.sh`
- Lint: `./lint.sh`
- Run tests: `python3 -m pytest`

TODO
----
- [x] MNIST (JAX)
- [ ] Resnet (JAX)
- [ ] grad(MNIST) (JAX)
- [ ] grad(Resnet) (JAX)
- [ ] MNIST (TensorFlow)
- [ ] Resnet50 (TensorFlow)
- [ ] `brax <https://github.com/google/brax>`_
- [ ] General reduction support (not using ReduceSum/ReduceMax)
