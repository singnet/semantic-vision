#ifndef MATCHINGAPI_GETBOW_H
#define MATCHINGAPI_GETBOW_H



#include "Includes.h"
#include "fromResponseFunctions.h"
#include "DetectInterface.h"
#include "FeaturesInterface.h"

void getBowVoc(vector<Mat> baseDescs, Mat descs, Mat &vocabulary, int clusterCount, vector<Mat> * processedBase, Mat &processedBaseMat);
vector<vector<DMatch>> processOneDesc(Mat oneDesc, Mat vocabulary, Mat processedBase, int num);


#endif //MATCHINGAPI_GETBOW_H