FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG language

ENV SINGNET_REPOS=/opt/singnet
ENV GOPATH=${SINGNET_REPOS}/go
ENV PATH=${GOPATH}/bin:${PATH}

RUN mkdir -p ${GOPATH}

RUN apt-get update && \
    apt-get install -y \
    apt-utils \
    nano \
    vim \
    git \
    wget \
    curl \
    zip \
    libudev-dev \
    libusb-1.0-0-dev

RUN apt-get install -y python3 python3-pip

RUN pip3 install snet-cli

RUN SNETD_VERSION=`curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")'` && \
    cd /tmp && \
    wget https://github.com/singnet/snet-daemon/releases/download/${SNETD_VERSION}/snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    mv snet-daemon-${SNETD_VERSION}-linux-amd64/snetd /usr/bin/snetd

RUN cd ${SINGNET_REPOS} && \
    git clone --depth 1 https://github.com/singnet/dev-portal.git

RUN     apt-get install -y build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev; \
        git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc; \
        cd grpc; \
        git submodule update --init; \
        make; \
        make install; \
        apt-get install -y openjdk-8-jdk; \
        echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list; \
        curl https://bazel.build/bazel-release.pub.gpg | apt-key add -; \
        apt-get update; \
        apt-get install -y bazel; \
        apt-get upgrade -y bazel; \
        bazel build :all; \
        make install; \
        cd third_party/protobuf; \
        make; \
        make install;

RUN apt -y remove x264 libx264-dev && \
    apt -y install build-essential checkinstall cmake pkg-config yasm \
    git gfortran \
    libjpeg8-dev libpng-dev \
    software-properties-common

RUN apt -y install python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install grpcio
RUN python3 -m pip install grpcio-tools

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt -y update
RUN apt-get update
RUN apt -y install libjasper1 \
    libjasper-dev \
    libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
    libxine2-dev libv4l-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev libtbb-dev qt5-default \
    libatlas-base-dev \
    libfaac-dev libmp3lame-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libavresample-dev \
    x264 v4l-utils \
    libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev \
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
    cmake \
    unzip \
    git

RUN apt-get install -y wget

RUN apt-get install -y vim

RUN cd \
    && wget https://github.com/opencv/opencv/archive/4.0.0.zip \
    && unzip 4.0.0.zip \
    && cd opencv-4.0.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j4 \
    && make install \
    && cd \
&& rm 4.0.0.zip

RUN cd \
    && wget https://github.com/opencv/opencv_contrib/archive/4.0.0.zip \
    && unzip 4.0.0.zip \
    && cd opencv-4.0.0/build \
    && cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.0.0/modules/ .. \
    && make -j4 \
    && make install \
    && cd ../.. \
&& rm 4.0.0.zip

RUN pip3 install opencv-contrib-python \
    && pip3 install tensorflow-gpu \
    && pip3 install matplotlib \
    && pip3 install tqdm

RUN cd \
    && git clone https://github.com/rpautrat/SuperPoint.git \
    && cd SuperPoint \
    && pip3 install -e . \
    && echo "DATA_PATH = './'" >> ./superpoint/settings.py \
    && echo "EXPER_PATH = '../models/saved_models/'" >> ./superpoint/settings.py \
    && cd

WORKDIR ${SINGNET_REPOS}
