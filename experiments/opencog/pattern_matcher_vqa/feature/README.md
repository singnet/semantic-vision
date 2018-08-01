# How to build bottom-up-attention dependency

## Python 2

Create conda environment (pmvqa2):

```
conda create --name pmvqa2 python=2.7
source activate pmvqa2
conda install opencv=3.1.0 atlas bokeh ca-certificates certifi cffi click cloudpickle cudatoolkit cudnn cycler cython cytoolz dask dask-core dbus decorator distributed expat fontconfig freetype gflags glib glog gst-plugins-base gstreamer h5py hdf5 heapdict icu imageio intel-openmp jbig jinja2 jpeg leveldb libedit libffi libgcc libgcc-ng libgfortran-ng libiconv libpng libprotobuf libstdcxx-ng libtiff libxcb libxml2 lmdb locket markupsafe matplotlib mkl msgpack-python nccl ncurses networkx ninja numpy opencv openssl packaging pandas partd pcre pillow pip protobuf psutil pycparser pyparsing pyqt python python-dateutil pytorch pytz pywavelets pyyaml qt readline scikit-image scipy setuptools sip six snappy sortedcontainers sqlite tblib tk toolz tornado wheel xz yaml zict zlib
conda install -c conda-forge jpype1
pip install easydict
```

Build caffe:

```
git clone https://github.com/peteanderson80/bottom-up-attention.git
cd bottom-up-attention
patch -p 1 < ../bottom-up-attention.2.patch
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:$LD_LIBRARY_PATH
cd caffe
cp ../../Makefile.config .
make
make pycaffe
cd ../lib
export PYTHONPATH=$(pwd):$(pwd)/../caffe/python:$PYTHONPATH
make
```

## Python 3

Create conda environment (pmvqa3):

```
conda create --name pmvqa3 python=3
source activate pmvqa3
conda install opencv=3.1.0 atlas bokeh ca-certificates certifi cffi click cloudpickle cudatoolkit cudnn cycler cython cytoolz dask dask-core dbus decorator distributed expat fontconfig freetype gflags glib glog gst-plugins-base gstreamer h5py hdf5 heapdict icu imageio intel-openmp jbig jinja2 jpeg leveldb libedit libffi libgcc libgcc-ng libgfortran-ng libiconv libpng libprotobuf libstdcxx-ng libtiff libxcb libxml2 lmdb locket markupsafe matplotlib mkl msgpack-python nccl ncurses networkx ninja numpy opencv openssl packaging pandas partd pcre pillow pip protobuf psutil pycparser pyparsing pyqt python python-dateutil pytorch pytz pywavelets pyyaml qt readline scikit-image scipy setuptools sip six snappy sortedcontainers sqlite tblib tk toolz tornado wheel xz yaml zict zlib
conda install -c conda-forge jpype1
pip install easydict
```

Build caffe:

```
git clone https://github.com/peteanderson80/bottom-up-attention.git
cd bottom-up-attention
patch -p 1 < ../bottom-up-attention.3.patch
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:$LD_LIBRARY_PATH
cd caffe
cp ../../Makefile.config .
make
make pycaffe
cd ../lib
export PYTHONPATH=$(pwd):$(pwd)/../caffe/python:$PYTHONPATH
make
```
