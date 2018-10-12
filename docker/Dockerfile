#
# Docker file that builds RelEx and visual question answering pipeline.
#
# To start:
#    docker run -it -p 8888:8888 relex-vqa
#
# To demo:
#    open in browser localhost:8888
#    use password: password 
#
# That will open jupyter notebook with vqa demo
#
FROM ubuntu:18.04
# use bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# Avoid triggering apt-get dialogs (which may lead to errors). See:
# https://stackoverflow.com/questions/25019183/docker-java7-install-fail
ENV DEBIAN_FRONTEND noninteractive

ENV JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64

RUN apt-get update ; apt-get -y upgrade ; apt-get -y autoclean

# Java
RUN apt-get -y install maven screen telnet netcat-openbsd byobu \
                       wget vim git unzip sudo apt-utils

# incorrect packaging in ubuntu-xenial
# dpkg: error processing archive /var/cache/apt/archives/openjdk-9-jdk_9~b114-0ubuntu1_amd64.deb (--unpack):
# trying to overwrite '/usr/lib/jvm/java-9-openjdk-amd64/include/linux/jawt_md.h', which is also in package openjdk-9-jdk-headless:amd64 9~b114-0ubuntu1
RUN apt-get -o Dpkg::Options::="--force-overwrite" -y install openjdk-11-jdk

# GCC and basic build tools
RUN apt-get -y install gcc g++ make swig ant

# Wordnet
RUN apt-get -y install wordnet wordnet-dev wordnet-sense-index

# There are UTF8 chars in the Java sources, and the RelEx build will
# break if build in a C environment.
RUN apt-get -y install locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir /usr/local/share/java

WORKDIR /home/Downloads/

# JWNL - Never changes, so do this first.
RUN wget https://downloads.sourceforge.net/project/jwordnet/jwnl/JWNL%201.4/jwnl14-rc2.zip \
    && unzip jwnl14-rc2.zip; cd jwnl14-rc2; \
    cp jwnl.jar /usr/local/share/java/; \
    chmod 755 /usr/local/share/java/jwnl.jar; \
    cd ..; rm -rf jwnl*

# OpenNLP - Never changes, so do this first.
RUN wget https://archive.apache.org/dist/opennlp/opennlp-1.5.3/apache-opennlp-1.5.3-bin.tar.gz \
    && tar -zxf apache-opennlp-1.5.3-bin.tar.gz ;\
    cd apache-opennlp-1.5.3; \
    cp lib/*.jar /usr/local/share/java/; \
    cp lib/*.jar /usr/share/java/; \
    cp lib/opennlp-tools-1.5.3.jar /usr/local/share/java/opennlp-tools-1.5.0.jar; \
    cd .. ; rm -rf apache-opennlp*
# build tools
RUN apt install -y libboost-python-dev libblas-dev libboost-thread-dev libboost-filesystem-dev libboost-system-dev
RUN apt install -y cmake libboost-program-options-dev libboost-regex-dev libiberty-dev guile-2.2-dev protobuf-compiler uuid-dev



# Conda
# install conda 

RUN apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1

# Create and switch user. The user is privileged, with no password
# required.  That is, you can use sudo.
RUN adduser --disabled-password --gecos "ReLex USER" relex
RUN adduser relex sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER relex

RUN sudo chown -R relex:relex .
RUN sudo chown -R relex:relex /home/relex 
# Punch out ports
## plain-text-server.sh port
EXPOSE 3333
## opencog-server.sh port
EXPOSE 4444
## link-grammar-server.sh port
EXPOSE 9000

WORKDIR /home/relex

# Link Parser -- changes often
# Download the current released version of link-grammar.
# The wget gets the latest version w/ wildcard

RUN wget -r --no-parent -nH --cut-dirs=2 https://www.abisource.com/downloads/link-grammar/current/
RUN tar -zxf current/link-grammar-5*.tar.gz
# get linkgrammar version
RUN bash -l -c 'echo `ls|grep link|sed 's/link-grammar-//g'` >> LINKGRAMMAR_VERSION'

USER root

RUN cd link-grammar-5.*/; ./configure; make -j6; sudo make install; \
    ldconfig;

USER relex
RUN cd link-grammar-5.*/; mvn install:install-file \
    -Dfile=./bindings/java/linkgrammar-`cat ../LINKGRAMMAR_VERSION`.jar \
    -DgroupId=org.opencog \
    -DartifactId=linkgrammar \
    -Dversion=`cat ../LINKGRAMMAR_VERSION` \
    -Dpackaging=jar

USER root
RUN rm -rf * link-grammar*

USER relex

# Relex -- changes often
RUN wget https://github.com/opencog/relex/archive/master.tar.gz -O master.tar.gz
RUN tar -xvf master.tar.gz; cd relex-master; mvn install



RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib/" >> /home/relex/.profile
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/relex/miniconda3/envs/pmvqa3/lib/" >> /home/relex/.profile

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b && \
    rm ~/anaconda.sh

RUN /home/relex/miniconda3/bin/conda create -y --name pmvqa3 python=3.5 && \
source /home/relex/miniconda3/bin/activate pmvqa3 && echo 'opencv ==3.1.0' > $(dirname $(which python))/../conda-meta/pinned && \
    conda install -y opencv=3.1.0 atlas bokeh ca-certificates certifi cffi click cloudpickle cudatoolkit cudnn cycler cython cytoolz dask dask-core dbus decorator distributed expat fontconfig freetype gflags glib glog gst-plugins-base gstreamer h5py hdf5 heapdict icu imageio intel-openmp jbig jinja2 jpeg leveldb libedit libffi libgcc libgcc-ng libgfortran-ng libiconv libpng libprotobuf libstdcxx-ng libtiff libxcb libxml2 lmdb locket markupsafe matplotlib mkl msgpack-python nccl ncurses networkx ninja numpy opencv openssl packaging pandas partd pcre pillow pip protobuf psutil pycparser pyparsing pyqt python python-dateutil pytorch pytz pywavelets pyyaml qt readline scikit-image scipy setuptools sip six snappy sortedcontainers sqlite tblib tk toolz tornado wheel xz yaml zict zlib jupyter && conda install -y -c conda-forge jpype1 && pip install easydict


# vqa piplie(java)

RUN wget https://github.com/singnet/semantic-vision/archive/master.zip

RUN unzip master.zip

RUN rm master.zip

RUN mkdir ~/projects

RUN mv semantic-vision-master/ ~/projects/semantic-vision-1

ENV JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64

RUN echo "export JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64" >> /home/relex/.profile

RUN cd ~/projects/semantic-vision-1/experiments/opencog/question2atomese && mvn package

# faster-r-cnn
RUN source /home/relex/miniconda3/bin/activate pmvqa3 && cd ~/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/feature && \
git clone https://github.com/peteanderson80/bottom-up-attention.git && \
cd bottom-up-attention && \
patch -p 1 < ../bottom-up-attention.3.patch && \
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:$LD_LIBRARY_PATH && \
cd caffe && \
cp ../../Makefile.config.3 Makefile.config && \
make ; make pycaffe ; cd ../lib && export PYTHONPATH=$(pwd):$(pwd)/../caffe/python:$PYTHONPATH; make

RUN echo "export PYTHONPATH=\$PYTHONPATH:~/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/feature/bottom-up-attention/lib:~/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/feature/bottom-up-attention/lib/../caffe/python" >> /home/relex/.profile


USER root 

# cogutil
RUN wget https://github.com/opencog/cogutil/archive/master.zip
RUN unzip master.zip && cd cogutil-master/ && mkdir build &&  cd build && cmake ..
RUN cd cogutil-master/build && make install
RUN rm -rf cogutil-master master.zip

# atomspace

USER relex 

RUN wget https://github.com/opencog/atomspace/archive/master.zip
RUN unzip master.zip 
RUN source /home/relex/miniconda3/bin/activate pmvqa3 && cd atomspace-master && mkdir build && cd build/ && cmake .. && make
USER root 
run cd atomspace-master/build && make install
RUN rm -rf atomspace-master/build master.zip
USER relex 
RUN mv atomspace-master /home/relex/projects/atomspace

# opencog

USER relex 
RUN wget https://github.com/opencog/opencog/archive/master.zip
RUN unzip master.zip
RUN source /home/relex/miniconda3/bin/activate pmvqa3 && cd opencog-master && mkdir build && cd build/ && cmake .. && make
USER root
RUN cd opencog-master/build && make install
RUN rm -rf opencog-master/build master.zip
USER relex
RUN mv opencog-master /home/relex/projects/opencog

# download some images
RUN mkdir -p /home/relex/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/images

RUN echo "export GUILE_AUTO_COMPILE=0" >> /home/relex/.profile

RUN echo "(use-modules (ice-9 readline)) (activate-readline)\ 
(add-to-load-path \"/usr/local/share/opencog/scm\")\
(add-to-load-path \"/home/relex/projects/opencog/examples/pln/conjunction/\")\
(add-to-load-path \"/home/relex/projects/atomspace/examples/rule-engine/rules/\")\
(add-to-load-path \"/home/relex/projects/opencog/opencog/pln/rules/\")\
(add-to-load-path \".\")\
(use-modules (opencog))\
(use-modules (opencog query))\
(use-modules (opencog exec))" >> /home/relex/.guile

RUN echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.5/dist-packages/" >> /home/relex/.profile
RUN wget  https://raw.githubusercontent.com/singnet/semantic-vision/dca3aed52578084e39bb52ffff79dbff11098827/experiments/opencog/pattern_matcher_vqa/vqa -O /home/relex/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/vqa
RUN chmod +x /home/relex/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/vqa

RUN source /home/relex/miniconda3/bin/activate pmvqa3 && jupyter notebook --generate-config 
RUN echo "c.NotebookApp.password = 'sha1:b6e570f197d6:b920933b262cf450f5c11dc21d878c53972cb2fa'" >> /home/relex/.jupyter/jupyter_notebook_config.py

RUN echo "c.NotebookApp.password_required = False" >> /home/relex/.jupyter/jupyter_notebook_config.py

RUN echo "c.NotebookApp.port = 8888" >> /home/relex/.jupyter/jupyter_notebook_config.py

RUN chmod +x /home/relex/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/vqa

CMD /home/relex/projects/semantic-vision-1/experiments/opencog/pattern_matcher_vqa/vqa
