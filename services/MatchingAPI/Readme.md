# How to use MatchingAPI

## Prerequisites
Opencv and contrib modules. You can find it there.
**Opencv**
https://github.com/opencv/opencv

**Contrib modules**
https://github.com/opencv/opencv_contrib

**grpc and protobuf**

## Prerequisites using docker
In folder For_docker you can find file Dockerfile using which you'll get image which has all necessary prerequisites
and more. To build image run following command
   
    docker build -t image_name - < path/to/Dockerfile

This will create docker image using which you can run container as follows
    
    docker run -ti image_name bash
    
Add any necessary additional options to this command by yourself. Inside of docker container (after docker run) you can
git clone this repo and build it using build.sh as follows:

    chmod +x build.sh
    ./build.sh
    
After build.sh done, you can 
find folder build and Server and Client executable files. Client is currently showing only result of getTransformByImage
but you can launch some python examples in py_examples folder. If you need to change any parameters like detector name,
descriptor name etc you can edit them inside python example files getSomething.py or create your own file.

## Usage
Cpp usage can be located in Client.cpp. There you can find usage of all currently done functions. 
In folder py_examples are examples of getting something (descriptors, keypoints etc) in Python for this service.
 
See also Readme_service.md for further explanation of input parameters.

See README_Detectors_Api.md for possible detectors and their input parameters.
See README_Features_Api.md for possible descriptors and their input parameters.
See README_transform_Api.md for possible transformations and their input parameters.

By default, Woods.jpg and Woods2.jpg are used in examples and Client.cpp. Change it if you need to input some other 
images