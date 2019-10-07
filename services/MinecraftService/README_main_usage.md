#Main usage

### Building

Use Dockerfile in For_docker folder to build image with all the dependencies you need. Use build.sh to build service 
after that. It will generate necessary grpc files and download necessary archives with pre-trained models.

### Usage
Run server with python server.py and then run client.py to inference service. Check client.py to see how to use server.
You can inference server using included images or using your own images. Though, network works only with images less or 
equal to 512x512 and will resize any larger image to fit these restrictions. So if your image is large, output will be 
smaller.

### Description

Currently, service is using only CycleGan and UGATIT models (you can choose either of these to inference server) 
pre-trained to transform style of landscapes. Other models or other pre-trained datasets will occure in future if 
there will be any need for this from the community.

### References
We've used CycleGan from here https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
and UGATIT from here https://github.com/znxlwm/UGATIT-pytorch
We've trained these two models on handmade dataset using summer2winter_yosemite dataset (both summer and winter were 
mixed in the one folder and used as train A dataset) and minecraft videos from youtube, sliced to images and filtered 
from irrelevant images, as train B dataset. So both networks could possibly learn real-world images' distribution and 
minecraft images' distribution to transform one to another. Results are not so great, but there are some interesting 
pictures you could get for example with provided 5 images.