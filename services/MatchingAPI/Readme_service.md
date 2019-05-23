# Service. 
## How to use, what's inside

There are several different methods which are currently implemented in this Matching service. To learn about detectors and 
features please look into corresponding readmes. Parameters for descriptor and detector must be set as string of
"parameter_name value parameter_name2 value2" with spaces as delimeters.

### getKP
This one computes keypoints on the given image and returns them.

#### input:

##### image 
bytes in proto, implemented as string in grpcc

##### detector
string specifying detector which will be used

##### parameters
parameters for detector

##### response
If everything is fine and tuned on server side and on client side, here client will recieve keypoints and status
    
#### Output
As output client will recieve, as said earlier, response of type keypointResponse. This response could be used further 
to get descriptors for these keypoints and consists of following fields:

##### Status
String. Contains status of operation

##### Keypoints 
Keypoints of type google::protobuf::RepeatedPtrField< ::MatchingApi::keyPoint >&. It is actually a structure which 
consists of float size, float angle, float x, float y, float response, int octave, int class_id. All these fields are 
taken from the cv::KeyPoint structure and used to fulfill cv::KeyPoint in further usage

#### usage example:
    snet client call $ORGANIZATION_ID $SERVICE_ID getKP '{ "file@image": "Woods.jpg", "detector_name" : "ORB", "parameters" : "nfeatures 1000 scaleFactor 1.5 nlevels 7 edgeThreshold 100 firstLevel 0"  }'

you will receive something like that (but more keypoints of course):

    keypoints {
      size: 156.9375
      angle: 249.16946411132812
      x: 1599.75
      y: 602.4375
      response: 5.275914645608282e-06
      octave: 4
      class_id: -1
    }
    status: "Success"
    
### getDescByImage
As it could be seen from name, this one computes descriptor and requires image as input.

#### input

##### image
bytes in proto, implemented as string in grpcc.

##### descriptor
string specifying descriptor which will be used

##### descparams 
parameters for descriptor

##### detector 
string specifying detector which will be used

##### detparams 
parameters for detector

##### response
If everything is fine and tuned on server side and on client side, here client will recieve descriptor, status and keyponts.

#### Output
As output client will recieve, as said earlier, response of type descriptorResponse. This response could be used further 
to get matches and consists of following fields:

##### Status
String. Contains status of operation

##### keypoints
Keypoints which were detected. Since method takes only image as input and no keypoints, they are need to be detected 
inside. And, for any usage, client will get these keypoints alongside with descriptor. To see what's inside see getKP.

##### descriptor
Type google::protobuf::Descriptor*. It is actually a structure with the following fields: 
float onedescF, bytes (string) onedescU. Depending on the descriptor type (float or uchar) one of these fields will 
contain descriptor for keypoint. 

#### usage example
    snet client call $ORGANIZATION_ID $SERVICE_ID getDescByImage '{ "file@image": "Woods.jpg", "descriptor_name" : "ORB", "desc_parameters" : "WTA_K 4", "detector_name" : "ORB", "det_parameters" : "nfeatures 1000 scaleFactor 1.5 nlevels 7 edgeThreshold 100 firstLevel 0"  }'

Descriptor for ORB will look like this:
    features {
      onedescU: 120
      onedescU: 137
      onedescU: 36
      onedescU: 24
      onedescU: 139
      onedescU: 88
      onedescU: 153
      onedescU: 120
      onedescU: 33
      onedescU: 46
      onedescU: 16
      onedescU: 143
      onedescU: 225
      onedescU: 32
      onedescU: 29
      onedescU: 142
      onedescU: 86
      onedescU: 194
      onedescU: 171
      onedescU: 179
      onedescU: 108
      onedescU: 46
      onedescU: 23
      onedescU: 173
      onedescU: 233
      onedescU: 142
      onedescU: 245
      onedescU: 237
      onedescU: 96
      onedescU: 152
      onedescU: 203
      onedescU: 69
    }

### getDescByKp
This function computes descriptor of input keypoints.

#### input
Instead of detector and detector_params this function asks for keypointResponse as input. Just put what getKP gives you 
here that's all. 

##### inputKeypoints
type keypointResponse. This thing is an output of getKP. Just use it.

##### descriptorResponse
Same thing as in getDescByImage but keypoints are empty.

#### usage example
    Currently no usage through snet. Use ByImage version
    
### getMatch
This function matches two descriptors and outputs matches. Currently it's just simple brute force matcher.

#### input

##### inputFeatures_first, inputFeatures_second
Type descriptorResponse both or descriptorByKpsResponse both. Just put what getDescByImage or getDescByKp gives you.

##### mresponse
Output matches

#### output
Type matchingResponse. Contains following fields

##### status
String. Contains status of operation

##### all_matches
Array of google::protobuf::RepeatedPtrField< ::MatchingApi::matchedPoint >. It is a structure actually which consists of:
int queryIdx, int trainIdx, int imgIdx, float distance. Same as cv::DMatch and could be used to fullfill vector<DMatch>
to for example drawKeypoints

#### example of usage
Currently no usage through snet. use ByImage version instead.

### getMatchByImage

getMatch but without any need of pre-detected keypoints or descriptors

#### input

#### image_first / image_second
bytes in proto, implemented as string in grpcc.

##### descriptor_name
string specifying descriptor which will be used

##### desc_parameters 
parameters for descriptor

##### detector_name 
string specifying detector which will be used

##### det_parameters 
parameters for detector

#### output

#### all_matches
Array of google::protobuf::RepeatedPtrField< ::MatchingApi::matchedPoint >. It is a structure actually which consists of:
int queryIdx, int trainIdx, int imgIdx, float distance. Same as cv::DMatch and could be used to fullfill vector<DMatch>
to for example drawKeypoints

#### status
String. Contains status of operation

#### keypoints_first / keypoints_second
Keypoints which were detected. Since method takes only image as input, they are need to be detected inside. And, for any
usage, client will get these keypoints alongside with transform parameters. To see what's inside see getKP.

#### usage
    snet client call $ORGANIZATION_ID $SERVICE_ID getMatchByImage '{ "file@image_first": "../Woods.jpg", "file@image_second" : "../Woods2.jpg", "descriptor_name" : "ORB", "desc_parameters" : "WTA_K 4", "detector_name" : "ORB", "det_parameters" : "nfeatures 1000 scaleFactor 1.5 nlevels 7 edgeThreshold 100 firstLevel 0" }'

You will receive something like this:
    all_matches {
      queryIdx: 4
      trainIdx: 58
      distance: 336.1026611328125
    }

### getTransformParameters

This function returns parameters of an input transform given 2 images, detected keypoints and matches. 

#### input

transformRequest consists of:

##### all_matches 
Array of google::protobuf::RepeatedPtrField< ::MatchingApi::matchedPoint >. It is a structure actually which consists of:
int queryIdx, int trainIdx, int imgIdx, float distance.

##### transform_type
Type of the transform parameters of which you want to get

##### transform_input_parameters
Parameters for transform. Note that not every transform has parameters

##### keypoints_first / keypoints_second

Keypoints of corresponding images detected by getKP or by any of your own detectors. Type keypointResponse (see getKP 
output)

##### output

##### status

String. Contains status of operation

##### transform_parameters

Array with transform parameters for given points 

#### usage

Currently no usage through snet. Use ByImage version instead.

### getTransformParametersByImage

Well, it is the same getTransformParameters but using images. No keypoints and matches needed. But additional detector
name/parameters descriptor name/parameters are need to be given to function

#### input

##### image_first / image_second
bytes in proto, implemented as string in grpcc.

##### descriptor_name
string specifying descriptor which will be used

##### desc_parameters 
parameters for descriptor

##### detector_name 
string specifying detector which will be used

##### det_parameters 
parameters for detector

##### transform_type
Type of the transform parameters of which you want to get

##### transform_input_parameters
Parameters for transform. Note that not every transform has parameters

#### output
same as for getTransformParameters

#### usage
    snet client call $ORGANIZATION_ID $SERVICE_ID getTransformParametersByImage '{ "file@image_first": "../Woods.jpg", "file@image_second" : "../Woods2.jpg", "descriptor_name" : "ORB", "desc_parameters" : "WTA_K 4", "detector_name" : "ORB", "det_parameters" : "nfeatures 1000 scaleFactor 1.5 nlevels 7 edgeThreshold 100 firstLevel 0", "transform_type" : "Bilinear", "transform_input_parameters" : "" }'

You will receive something like that

transform_parameters: 252.20627262266166
transform_parameters: 0.011269062626934574
transform_parameters: 0.01949454307748601
transform_parameters: 0.0
transform_parameters: 1.1279451100671525e-06
transform_parameters: 0.0
transform_parameters: 256.33048083023186
transform_parameters: 0.014174183414889147
transform_parameters: 0.032841365983950634
transform_parameters: 0.0
transform_parameters: -3.340623914930427e-05
transform_parameters: 0.0


