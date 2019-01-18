FROM opencog/vqa

USER relex
RUN git clone https://github.com/singnet/semantic-vision.git

RUN source /home/relex/miniconda3/bin/activate pmvqa3 && \
    conda install pip && pip install grpcio-tools

RUN cd semantic-vision &&\
    source /home/relex/.profile &&\
    source /home/relex/miniconda3/bin/activate pmvqa3 &&\
    cd /home/relex/semantic-vision/services/vqa-service &&\
    python3 setup.py build && python3 setup.py install

RUN rm -r semantic-vision
RUN echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'" >> /home/relex/.profile
