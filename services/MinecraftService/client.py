import sys
import grpc

import imageio
import io
import numpy as np
import base64

from PIL import Image

import MinecraftizingService_pb2 as pb2
import MinecraftizingService_pb2_grpc as pb2_grpc

def base64_to_jpg(base64img, output_file_path=""):
    """Decodes from base64 to jpg. If output_file_path is defined, saves the decoded image."""

    decoded_jpg = base64.b64decode(base64img)
    jpg_bytes = io.BytesIO(decoded_jpg)
    image = Image.open(jpg_bytes)
    if output_file_path != "":
        # If image is PNG, convert to JPG
        if image.format == 'PNG':
            image = image.convert('RGB')
        image.save(output_file_path, format='JPEG')
    return decoded_jpg


def getMinecraftiziedImage(channel, imgpath, modelname, experiment_name):
    stub = pb2_grpc.MinecraftizingServiceStub(channel)
    request = pb2.minecraftRequest()
    with open(imgpath, 'rb') as f:
        request.input_image = f.read()
    request.network_name = modelname
    request.dataset = experiment_name
    response = stub.getMinecraftiziedImage(request)
    return response

def main():
    channel = grpc.insecure_channel('localhost:51001', options=[('grpc.max_send_message_length', -1), (
        'grpc.max_receive_message_length', -1)])

    response = getMinecraftiziedImage(channel, "2.jpg", "cycle_gan", "minecraft_landscapes")

    base64_to_jpg(response.output, "result.jpg")
    print(response.status)


if __name__ == '__main__':
    main()
