#!/usr/bin/env python3

import distutils.command.build_py as build_py
from distutils.core import setup
import shlex
import subprocess

def compile_proto():
    cmd = shlex.split('python3 -m grpc_tools.protoc -I. --python_out=vqaservice --grpc_python_out=vqaservice ./service.proto')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("Error compiling proto file: {0}".format(err))


class BuildPyCommand(build_py.build_py):
    """Custom build command."""

    def run(self):
        compile_proto()
        build_py.build_py.run(self)


setup(name='vqa',
      version='0.1',
      description='SingularityNet visual question answering service',
      cmdclass={'build_py': BuildPyCommand},
      author='Anatoly Belikov',
      author_email='abelikov@singularitynet.io',
      url='https://github.com/singnet/semantic-vision',
      scripts=['vqa_service.py', 'vqa_client.py'],
      packages=['vqaservice'],
      install_requires=[
          'protobuf',
          'grpcio-tools',
          'grpcio'],)
