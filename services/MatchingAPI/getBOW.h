#ifndef MATCHINGAPI_GETBOW_H
#define MATCHINGAPI_GETBOW_H

#include "fbow/fbow.h"
#include "fbow/vocabulary_creator.h"

#include "Includes.h"
#include "fromResponseFunctions.h"
#include "DetectInterface.h"
#include "FeaturesInterface.h"

using namespace fbow;

void getBowVoc(vector<Mat> baseDescs, Vocabulary * vocabulary, vector<fBow> * processedBase);
vector<pair<double, int>> processOneDesc(vector<fBow> processedBase, Mat oneDesc, Vocabulary &vocabulary, int num);


#endif //MATCHINGAPI_GETBOW_H