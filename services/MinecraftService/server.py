import sys
from concurrent import futures
import time

import grpc
import cv2
import imageio
import base64
import io
from PIL import Image
import numpy as np

import MinecraftizingService_pb2 as pb2
import MinecraftizingService_pb2_grpc as pb2_grpc

from CycleGanInference import getCycleGanTransform
from UGATITInference import getUGATITTransform

def jpg_to_base64(jpgimg, open_file=False):

    if open_file:
        try:
            jpgimg = Image.open(jpgimg)
        except Exception as e:
            log.error(e)
            raise
    imgbuffer = io.BytesIO()
    try:
        jpgimg.save(imgbuffer, format='JPEG')
    except Exception as e:
        log.error(e)
        raise
    imgbytes = imgbuffer.getvalue()
    return base64.b64encode(imgbytes)




class Minecraftizing(pb2_grpc.MinecraftizingServiceServicer):

    def getMinecraftiziedImage(self, request, context):
        if (request.network_name == 'cycle_gan'):
            transform_result = getCycleGanTransform(request.input_image, request.dataset)
        elif (request.network_name == 'UGATIT'):
            transform_result = getUGATITTransform(request.input_image, request.dataset)

        imageio.imwrite("temp.jpg", transform_result)
        response = pb2.minecraftResponse()
        response.output = jpg_to_base64("temp.jpg", open_file=True).decode('utf-8')
        response.status = "Image transformed using desired NN"
        return response

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_MinecraftizingServiceServicer_to_server(Minecraftizing(), server)
    server.add_insecure_port("[::]:51001")
    server.start()
    print("Server listening on 0.0.0.0:{}".format(51001))
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()
