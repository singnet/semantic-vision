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
    string image("path/image.extension");
    string detector("ORB");
    string detector_params("nfeatures 2000 nlevels 7");
    string image_bytes = getImageString(image); //getImageString is the function which will upload image from path to string of chars
    keypointResponse responsekp;
    string reply = client.getKP(image_bytes, detector, detector_params, &responsekp);
    
#### Snet usage example

    snet client call snet match-service getKP '{ "file@image": "Woods.jpg", "detector_name" : "ORB", "parameters" : "WTA_K 4"  }'
    
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
    string image("path/image.extension");
    string detector("ORB");
    string detector_params("nfeatures 2000 nlevels 7");
    string descriptor("ORB");
    string desc_params("WTA_K 4")
    string image_bytes = getImageString(image);
    descriptorResponse response;
    string reply = client.getDesc(image_bytes, descriptor, desc_params, detector, detector_params, &response);

#### Snet usage example
    snet client call snet match-service getDescByImage '{ "file@image": "Woods.jpg", "detector_name" : "ORB", "det_parameters" : "WTA_K 4", "descriptor_name" : "ORB", "desc_parameters" : ""  }'

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
    string image("path/image.extension");
    string detector("ORB");
    string detector_params("nfeatures 2000 nlevels 7");
    string image_bytes = getImageString(image); //getImageString is the function which will upload image from path to string of chars
    keypointResponse responsekp;
    string reply = client.getKP(image_bytes, detector, detector_params, &responsekp);
    string descriptor("ORB");
    string desc_params("WTA_K 4")
    descriptorResponse response_bykps;
    reply = client.getDescByKp(image_bytes2, descriptor, desc_params, responsekp, &response_bykps);
    
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
    matchingResponse mresponse;
    string reply = client.getMatch(response, response2, &mresponse); //response and response2 are outputs of getDescBy methods

#### example of what to do with matches
    vector<DMatch> matches;
    for (auto& oneMatch : mresponse.all_matches())
    {matches.emplace_back(oneMatch.queryidx(), oneMatch.trainidx(), oneMatch. imgidx(), oneMatch.distance());}

    Mat img_matches;
    Mat imageMat1 = imread(image);
    Mat imageMat2 = imread(image2);
    vector<KeyPoint> points = kpsFromResponse(responsekp1);
    vector<KeyPoint> points2 = kpsFromResponse(responsekp2);
    drawMatches(imageMat1, points, imageMat2, points2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey();

This simple code will draw matches which client will recieve from getMatches

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

    string image("../Woods.jpg");
    string image2("../Woods2.jpg");

    string detector("ORB");
    string descriptor("ORB");
    string detector_params("");
    string desc_params("");
    
    string image_bytes = getImageString(image);
    string image_bytes2 = getImageString(image2);
    
    matchingByImageResponse response;
    string status = getMatchByImage(image_bytes, image_bytes2, detector, detector_params, descriptor, desc_params, &response);

#### Snet usage
    snet client call snet match-service getMatchByImage '{ "file@image_first": "Woods.jpg", "file@image_second": "Woods2.jpg", "detector_name" : "ORB", "det_parameters" : "WTA_K 4", "descriptor_name" : "ORB", "desc_parameters" : ""  }'

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

    transformResponse reply;
    string transform_type = "Bilinear"
    string transform_input_parameters = "" //no parameters needed for that transform
    string status = getTransformParameters(transform_type, transform_input_parameters, first_kps, second_kps,  matches,
            &reply)

Possible transforms and parameters you can find in README_transform_Api.

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

    string image("../Woods.jpg");
    string image2("../Woods2.jpg");

    string detector("ORB");
    string descriptor("ORB");
    string detector_params("");
    string desc_params("");

    string image_bytes = getImageString(image);
    string image_bytes2 = getImageString(image2);
    string transf_type("Bilinear");
    string transf_params_in("");
    transformResponse transfReply;
    string status = client.getTransformParametersByImage(image_bytes, image_bytes2, detector, detector_params, 
        descriptor, desc_params, transf_type, transf_params_in, &transfReply);

#### Snet usage
    snet client call snet match-service getTransformParametersByImage '{ "file@image_first": "Woods.jpg", "file@image_second": "Woods2.jpg", "detector_name" : "ORB", "det_parameters" : "WTA_K 4", "descriptor_name" : "ORB", "desc_parameters" : ""  }'

### getClosestImage

This is the image retrieval algorithm. It will get you several closest images to the input image.

Simple database which can be used for this service is here https://yadi.sk/d/PpLotfo4ORikFw

#### input

##### input_image
As before, image in bytes.

##### image_base
Several images, type "repeated bytes".

##### descriptor_name, desc_parameters, detector_name, det_parameters
Same as before

##### numOfImagesToRetrieve
Number of closest images to numOfImagesToRetrieve

##### numOfClusters
Parameter for Bag of words clusterization. How much words will be in the bag.

#### output

##### images
repeated bytes of images. Size is equal to numOfImagesToRetrieve.

##### distances
repeated float of distances between images in descriptor space.

##### status
String. Contains status of operation

#### usage

See client.cpp -> main -> //getClosestImages usage