#ifndef MATCHINGAPI_ADDITIONALTRANSFORMATIONS_H
#define MATCHINGAPI_ADDITIONALTRANSFORMATIONS_H

#include "Includes.h"

void findShift(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transfParameters);

void findShiftScale(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transf_parameters);

void findShiftRot(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transf_parameters);

bool findPoly(vector<double> x, vector<double> y, vector<double> xi, vector<double> yi,  vector<double> * transform_params, bool is_bilin);

#endif //MATCHINGAPI_ADDITIONALTRANSFORMATIONS_H
