FROM opencog/vqa

USER relex
RUN git clone https://github.com/noskill/semantic-vision-1.git
RUN source /home/relex/miniconda3/bin/activate pmvqa3 && \
    conda install pip && pip install grpcio-tools
RUN cd semantic-vision-1 && git checkout -b vqa-service && git pull origin vqa-service &&\
    source /home/relex/.profile &&\
    source /home/relex/miniconda3/bin/activate pmvqa3 &&\
    cd /home/relex/semantic-vision-1/services/vqa-service &&\
    python3 setup.py build && python3 setup.py install

RUN echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'" >> /home/relex/.profile
