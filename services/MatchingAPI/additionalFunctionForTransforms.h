#ifndef MATCHINGAPI_ADDITIONALFUNCTIONFORTRANSFORMS_H
#define MATCHINGAPI_ADDITIONALFUNCTIONFORTRANSFORMS_H

#include "Includes.h"

#define SQR(a) ((a)*(a))
#define SQR_SUM(a,b) (SQR(a)+SQR(b))

#define NUM_POLY2_PARAMS 12

#define ZERO_VALUE      1E-8

#define ER_DETERM_ZERO 0
#define SUCCESSFUL 1


void poly2_addPoint(double *Matr, double *B, double x, double y, double xi, double yi, bool is_bilin);

bool SolveLinear(double *matrix, double *b, vector<double> * sol, int s);

void zeroEquations(double *Matr, double *B, int s);

double calcArrayAver(vector<double> x);

double calcArraysMultAver(vector<double> x, vector<double> y);

double calcArraysHomoAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2);

void calcFourArraysAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, double *Ax1, double *Ay1, double *Ax2, double *Ay2);

double calcArraysHeteroAver(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2);

void getShiftByRotScale(double Ax1, double Ay1, double Ax2, double Ay2, double scale, double angle, double *xsh, double *ysh);

double similar_sumErr(vector<double> x1, vector<double> y1, vector<double> x2, vector<double> y2, double scale, double angle, double xsh, double ysh);

#endif //MATCHINGAPI_ADDITIONALFUNCTIONFORTRANSFORMS_H
