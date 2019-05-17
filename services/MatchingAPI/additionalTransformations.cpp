#include "additionalTransformations.h"
#include "additionalFunctionForTransforms.h"



void findShift(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transfParameters)
{
    (*transfParameters).push_back(calcArrayAver(x2) - calcArrayAver(x1));
    (*transfParameters).push_back(calcArrayAver(y2) - calcArrayAver(y1));
}


void findShiftScale(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transf_parameters)
{
    double Ax1, Ay1, Ax2, Ay2, D3, D1;
    calcFourArraysAver(x1, y1, x2, y2, &Ax1, &Ay1, &Ax2, &Ay2);
    D3 = calcArraysHomoAver(x1, y1, x1, y1);
    D1 = calcArraysHomoAver(x1, y1, x2, y2);
    double scale = (Ax1 * Ax2 + Ay1 * Ay2 - D1) / (Ax1 * Ax1 + Ay1 * Ay1 - D3);
    (*transf_parameters).push_back(Ax2 - scale * Ax1);
    (*transf_parameters).push_back(Ay2 - scale * Ay1);
    (*transf_parameters).push_back(scale);
}


void findShiftRot(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, vector<double> * transf_parameters)
{
    double alp1, alp2, Ax1, Ay1, Ax2, Ay2, D1, D2, shx1, shy1, shx2, shy2, err1, err2;
    calcFourArraysAver(x1, y1, x2, y2, &Ax1, &Ay1, &Ax2, &Ay2);
    D1 = calcArraysHomoAver(x1, y1, x2, y2);
    D2 = calcArraysHeteroAver(x1, y1, x2, y2);
    alp1 = atan((-Ax1 * Ay2 + Ax2 * Ay1 + D2) / (Ax1 * Ax2 + Ay1 * Ay2 - D1));
    alp2 = alp1 + M_PI;
    getShiftByRotScale(Ax1, Ay1, Ax2, Ay2, 1.0, alp1, &shx1, &shy1);
    getShiftByRotScale(Ax1, Ay1, Ax2, Ay2, 1.0, alp2, &shx2, &shy2);
    err1 = similar_sumErr(x1, y1, x2, y2, 1.0, alp1, shx1, shy1);
    err2 = similar_sumErr(x1, y1, x2, y2, 1.0, alp2, shx2, shy2);
    if (err1 < err2)
    {
        (*transf_parameters).push_back(shx1);
        (*transf_parameters).push_back(shy1);
        (*transf_parameters).push_back(alp1);
    }
    else
    {
        (*transf_parameters).push_back(shx2);
        (*transf_parameters).push_back(shy2);
        (*transf_parameters).push_back(alp2);
    }
}

bool findPoly(vector<double> x, vector<double> y, vector<double> xi, vector<double> yi,  vector<double> * transform_params, bool is_bilin)
{
    int n = x.size(), i;
    double Matr[NUM_POLY2_PARAMS * NUM_POLY2_PARAMS], B[NUM_POLY2_PARAMS];
    zeroEquations(Matr, B, NUM_POLY2_PARAMS);
    for(i = 0; i < n; i++)
    {
        poly2_addPoint(Matr, B, x[i], y[i], xi[i], yi[i], is_bilin);
    }
    return SolveLinear(Matr, B, transform_params, NUM_POLY2_PARAMS);
}