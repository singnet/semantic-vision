import grpc

import MatchingAPI_pb2
import MatchingAPI_pb2_grpc

def transf_parameters_from_response(response):
    transform_parameters = []
    for param in response.transform_parameters:
        transform_parameters.append(param)
    return transform_parameters

def main():
    image_path_1 = '../Woods.jpg'
    image_path_2 = '../Woods2.jpg'
    detector = 'ORB'
    detector_params = 'nfeatures 1000 scaleFactor 5'
    descriptor = 'ORB'
    descriptor_params = ''
    transform = 'Bilinear'
    transform_parameters = ''
    channel = grpc.insecure_channel('localhost:32', options=[('grpc.max_send_message_length', -1), (
        'grpc.max_receive_message_length', -1)])
    service = MatchingAPI_pb2_grpc.MatchApiStub(channel)
    request = MatchingAPI_pb2.transformByImageRequest()
    with open(image_path_1, 'rb') as f:
        request.image_first = f.read()
    with open(image_path_2, 'rb') as f:
        request.image_second = f.read()
    request.detector_name = detector
    request.det_parameters = detector_params
    request.descriptor_name = descriptor
    request.desc_parameters = descriptor_params
    request.transform_type = transform
    request.transform_input_parameters = transform_parameters
    response = service.getTransformParametersByImage(request)
    print(response.status)
    print(transf_parameters_from_response(response))


if __name__ == "__main__":
    main()
