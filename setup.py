# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()


setup(
    name='xla2onnx',
    version='0.1.0',
    description='Convert XLA to ONNX',
    long_description=readme,
    author='Akira Kawata',
    author_email='akawashiro@users.noreply.github.com',
    url='https://github.com/akawashiro/xla2onnx',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs'))
)
