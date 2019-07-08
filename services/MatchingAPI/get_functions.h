
#ifndef MATCHINGAPI_GET_FUNCTIONS_H
#define MATCHINGAPI_GET_FUNCTIONS_H

#include "Includes.h"

string getKeypoint(string image, string detector, string detector_parameters, vector<KeyPoint> * output);

string getDescriptorByImage(string image, string detector, string detector_parameters, string descriptor,
                                     string descriptor_parameters, vector<vector<float>> *resultF,
                                     vector<vector<int>> *resultU, vector<KeyPoint> *kps);

string getDescriptorByKps(string image, string descriptor, string descriptor_parameters, vector<KeyPoint> &kps,
                                    vector<vector<float>> *resultF, vector<vector<int>> *resultU);

string getMatches(Mat desc1, Mat desc2, vector<DMatch>* matches);

string getMatchesByImg(string image1, string image2, string detector, string detector_parameters, string descriptor,
                       string descriptor_parameters, vector<KeyPoint> *kps1, vector<KeyPoint> *kps2, vector<DMatch>* matches);

string getTransformParams(string transformType, string transform_input_parameters, vector<DMatch> matches_in,
        vector<KeyPoint> first_kps, vector<KeyPoint> second_kps, vector<double> * transform_parameters);

string getTransformParamsByImg(string image1, string image2, string detector, string detector_parameters,
        string descriptor, string descriptor_parameters, string transformType, string transform_input_parameters,
        vector<double> * transform_parameters);

#endif //MATCHINGAPI_GET_FUNCTIONS_H

